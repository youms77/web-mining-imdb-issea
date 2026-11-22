"""Phase 6, extension EXPLORATOIRE : thèmes des phrases croisés avec le sentiment des phrases.

Limites à garder en tête :
- Le corpus n'a aucune étiquette par phrase : aucune exactitude par phrase n'est calculée ni publiée.
- Le classifieur est calibré sur des critiques entières : ses probabilités ne sont pas validées sur
  des phrases isolées. Le seul contrôle possible est l'accord avec VADER, autre méthode imparfaite.
- Les critiques de l'échantillon viennent de l'entraînement. Le modèle gelé les a donc déjà vues :
  les scores utilisent une copie de sa recette, réapprise sur le reste de l'entraînement.
  Le modèle gelé n'est ni modifié ni réécrit.

Exemple d'utilisation :
    python -m src.bertopic_phrases --etape scores
    python -m src.bertopic_phrases --etape aspects
    python -m src.bertopic_phrases --etape croiser
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score

from src import bertopic_model as bm
from src import features as f
from src import models as m
from src import topics as tp
from src.preprocessing import RACINE, pipeline_b

CHEMIN_GELE = RACINE / "models" / "sentiment_svm_lineaire.joblib"
CHEMIN_CHOIX = RACINE / "reports" / "phase05_choix_figes.json"
CHEMIN_TEXTES_SANS_NOTES = RACINE / "data" / "interim" / "phase5" / "textes_sans_notes.parquet"
CHEMIN_CLONE = bm.DOSSIER / "svm_hors_echantillon.joblib"
CHEMIN_CLONE_INFO = bm.DOSSIER / "svm_hors_echantillon.json"
CHEMIN_SCORES = bm.DOSSIER / "phrases_scores.parquet"
CHEMIN_ASPECTS = bm.DOSSIER / "phrases_aspects.parquet"
CHEMIN_GRILLE = bm.DOSSIER / "phrases_grille.json"
CHEMIN_CROISEMENT = bm.DOSSIER / "phrases_croisement.json"

GRILLE_TAILLE_MIN = (40, 80, 160, 320)
K_ASPECTS = 12
N_MOTS_B_MIN = 2  # une phrase sans contenu après le pipeline B ne peut pas décrire un thème
SEUIL_VADER = 0.05  # seuil usuel de la documentation de VADER pour un texte neutre
SEUILS_SENSIBILITE = (0.10, 0.90)
N_BOOTSTRAP = 1000
N_PROCESSUS = 3
CATEGORIES = ("negative", "incertaine", "positive")


def seuils_figes() -> tuple[float, float]:
    """Zone d'incertitude figée en phase 5, sur des critiques entières."""
    zone = json.loads(CHEMIN_CHOIX.read_text(encoding="utf-8"))["zone_incertitude"]
    return float(zone["seuil_bas"]), float(zone["seuil_haut"])


def categoriser(probabilites: np.ndarray, bas: float, haut: float) -> np.ndarray:
    """0 = négative, 1 = incertaine, 2 = positive. Les bornes restent dans la zone incertaine."""
    p = np.asarray(probabilites)
    return np.where(p < bas, 0, np.where(p > haut, 2, 1)).astype(np.int8)


def entrainer_clone() -> object:
    """Réapprend la recette gelée sur l'entraînement privé des 5 000 critiques de l'échantillon."""
    if CHEMIN_CLONE.exists():
        return joblib.load(CHEMIN_CLONE)
    echantillon = bm.construire_echantillon()
    id_train, id_test = f.charger_index()
    exclus = set(echantillon["id_brut"].tolist())
    if exclus & set(id_test.tolist()):
        raise RuntimeError("L'échantillon contient une critique de test.")
    textes = pd.read_parquet(CHEMIN_TEXTES_SANS_NOTES).set_index("id_brut")
    if set(textes.index.tolist()) & set(id_test.tolist()):
        textes = textes.loc[textes.index.isin(id_train)]
    etiquettes = f.charger_entrainement(("id_brut", "sentiment")).set_index("id_brut")["sentiment"]
    ids = np.array([i for i in np.sort(id_train) if i not in exclus])
    y = (etiquettes.loc[ids] == "positive").astype(int).to_numpy()
    recette = clone(m.charger_modele(CHEMIN_GELE)).set_params(n_jobs=N_PROCESSUS)
    debut = time.perf_counter()
    recette.fit(np.array(textes.loc[ids, "texte"].tolist(), dtype=object), y)
    bm.ecrire_json(CHEMIN_CLONE_INFO, {
        "critiques_apprises": int(len(ids)), "part_positive": float(y.mean()),
        "duree_s": round(time.perf_counter() - debut, 1),
        "critiques_de_test_utilisees": 0, "critiques_de_l_echantillon_utilisees": 0,
    })
    joblib.dump(recette, CHEMIN_CLONE, compress=3)
    return recette


def _vader(textes: list[str]) -> np.ndarray:
    return m.ReferenceVader().scores(textes)


def calculer_scores() -> pd.DataFrame:
    """Prépare les phrases et calcule les trois scores : copie hors échantillon, modèle gelé, VADER."""
    if CHEMIN_SCORES.exists():
        return pd.read_parquet(CHEMIN_SCORES)
    table = pd.read_parquet(bm.CHEMIN_PHRASES)
    phrases = table["phrase"].tolist()
    table["texte_a"] = Parallel(n_jobs=N_PROCESSUS, batch_size=500)(delayed(m.preparer_texte)(p) for p in phrases)
    table["texte_b"] = Parallel(n_jobs=N_PROCESSUS, batch_size=200)(delayed(pipeline_b)(p) for p in phrases)
    textes = np.array(table["texte_a"].tolist(), dtype=object)
    table["proba_hors_echantillon"] = entrainer_clone().predict_proba(textes)[:, 1]
    table["proba_modele_gele"] = m.charger_modele(CHEMIN_GELE).predict_proba(textes)[:, 1]
    morceaux = [phrases[i:i + 2000] for i in range(0, len(phrases), 2000)]
    table["vader"] = np.concatenate(Parallel(n_jobs=N_PROCESSUS)(delayed(_vader)(x) for x in morceaux))
    echantillon = bm.construire_echantillon().set_index("id_brut")["y"]
    table["y_critique"] = echantillon.loc[table["id_brut"]].to_numpy()
    table.to_parquet(CHEMIN_SCORES, index=False)
    return table


def ajuster_aspects() -> pd.DataFrame:
    """Topics des phrases : grille HDBSCAN avec la règle des critiques, puis réduction à `K_ASPECTS`."""
    if CHEMIN_ASPECTS.exists():
        return pd.read_parquet(CHEMIN_ASPECTS)
    table = calculer_scores()
    embeddings = bm.calculer_embeddings_phrases()
    garder = (table["texte_b"].str.split().str.len() >= N_MOTS_B_MIN).to_numpy()
    documents = table.loc[garder, "texte_b"].tolist()
    vecteurs = embeddings[garder]
    debut = time.perf_counter()
    coordonnees = bm.calculer_umap(vecteurs, bm.GRAINE, bm.DOSSIER / "umap_phrases.npy")
    duree_umap = time.perf_counter() - debut
    termes = tp.charger_matrices_topics()["termes"]
    lignes, modeles, groupes = [], {}, {}
    for taille in GRILLE_TAILLE_MIN:
        debut = time.perf_counter()
        modele, sujets = bm.ajuster_bertopic(documents, vecteurs, coordonnees, termes, taille)
        mots = bm.mots_topics(modele)
        groupes[str(taille)] = mots
        modeles[taille] = modele
        lignes.append({
            "taille_min": taille, "n_topics": len(mots), "part_hors_topic": float((sujets == -1).mean()),
            "plus_grand_topic": float(pd.Series(sujets[sujets != -1]).value_counts(normalize=True).iloc[0]),
            "diversite": tp.diversite(mots), "duree_s": round(time.perf_counter() - debut, 1),
        })
        print(lignes[-1], flush=True)
    reference = [t.split() for t in f.charger_entrainement(("id_brut", "texte_b"))["texte_b"]]
    mesures = bm.coherences_groupees(groupes, reference)
    for ligne in lignes:
        ligne["c_v_entrainement"] = float(np.mean(mesures[str(ligne["taille_min"])]["c_v"]))
        ligne["c_npmi_entrainement"] = float(np.mean(mesures[str(ligne["taille_min"])]["c_npmi"]))
    choix = bm.choisir_taille_min(lignes)
    reduit, sujets = bm.reduire_a_k(modeles[choix["taille_min"]], documents, K_ASPECTS)
    mots = bm.mots_topics(reduit)
    coherence = bm.coherences_groupees({"aspects": mots}, reference)["aspects"]
    bm.ecrire_json(CHEMIN_GRILLE, {
        "regle": bm.REGLE_TAILLE_MIN, "grille": lignes, "choix": choix, "k_aspects": K_ASPECTS,
        "phrases_modelisees": int(garder.sum()), "phrases_sans_contenu_ecartees": int((~garder).sum()),
        "duree_umap_s": round(duree_umap, 1), "mots": mots,
        "c_v_entrainement_aspects": float(np.mean(coherence["c_v"])),
        "c_v_par_aspect": coherence["c_v"], "diversite_aspects": tp.diversite(mots),
        "part_hors_topic_aspects": float((sujets == -1).mean()),
    })
    sortie = table.loc[garder].copy()
    sortie["aspect"] = sujets
    sortie.to_parquet(CHEMIN_ASPECTS, index=False)
    return sortie


def _ic_grappes(comptes: np.ndarray, totaux: np.ndarray, tirages: np.ndarray) -> tuple[float, float]:
    """IC à 95 % d'une proportion : les critiques entières sont rééchantillonnées avec remise."""
    numerateur, denominateur = tirages @ comptes, tirages @ totaux
    parts = 100 * numerateur / np.maximum(denominateur, 1)
    return float(np.percentile(parts, 2.5)), float(np.percentile(parts, 97.5))


def mesurer_accord_vader(probabilites: np.ndarray, vader: np.ndarray, bas: float, haut: float) -> dict:
    """Accord entre le classifieur et VADER. Ce n'est PAS une exactitude : aucun des deux n'est une vérité."""
    non_neutres = np.abs(vader) >= SEUIL_VADER
    signe_vader = (vader[non_neutres] > 0).astype(int)
    signe_modele = (probabilites[non_neutres] >= 0.5).astype(int)
    tranche = non_neutres & ((probabilites < bas) | (probabilites > haut))
    return {
        "phrases": int(len(vader)),
        "part_vader_neutre": float((~non_neutres).mean()),
        "correlation_spearman_toutes_phrases": float(stats.spearmanr(probabilites, vader).statistic),
        "accord_brut_vader_non_neutre": float((signe_vader == signe_modele).mean()),
        "kappa_cohen_vader_non_neutre": float(cohen_kappa_score(signe_vader, signe_modele)),
        "part_phrases_tranchees_par_les_deux": float(tranche.mean()),
        "accord_brut_phrases_tranchees": float(
            ((vader[tranche] > 0).astype(int) == (probabilites[tranche] >= 0.5).astype(int)).mean()
        ),
    }


def part_critiques_avec_phrase(table: pd.DataFrame, colonne: str, classe: int, sens: str,
                               bas: float, haut: float) -> float:
    """Part des critiques d'une classe ayant au moins une phrase classée dans le sens opposé."""
    sous = table.loc[table["y_critique"] == classe]
    if sens == "negative":
        marque = sous[colonne] < bas
    else:
        marque = sous[colonne] > haut
    return float(marque.groupby(sous["id_brut"]).any().mean())


def croiser() -> dict:
    """Répartition des phrases de chaque aspect selon leur sentiment et la classe de la critique."""
    if CHEMIN_CROISEMENT.exists():
        return json.loads(CHEMIN_CROISEMENT.read_text(encoding="utf-8"))
    bas, haut = seuils_figes()
    toutes = calculer_scores()
    table = ajuster_aspects()
    table["categorie"] = categoriser(table["proba_hors_echantillon"].to_numpy(), bas, haut)
    ids = np.sort(table["id_brut"].unique())
    rang = pd.Series(np.arange(len(ids)), index=ids)
    table["rang_critique"] = rang.loc[table["id_brut"]].to_numpy()
    y_par_critique = table.groupby("rang_critique")["y_critique"].first().to_numpy()
    generateur = np.random.default_rng(bm.GRAINE)
    tirages = np.vstack([
        np.bincount(generateur.integers(0, len(ids), len(ids)), minlength=len(ids))
        for _ in range(N_BOOTSTRAP)
    ])

    aspects = []
    for aspect in sorted(a for a in table["aspect"].unique() if a != -1):
        sous = table.loc[table["aspect"] == aspect]
        ligne = {"aspect": int(aspect), "phrases": int(len(sous)), "part_des_phrases": float(len(sous) / len(table))}
        for classe, nom in ((1, "critiques_positives"), (0, "critiques_negatives")):
            groupe = sous.loc[sous["y_critique"] == classe]
            totaux = np.bincount(groupe["rang_critique"], minlength=len(ids))
            bloc = {"phrases": int(len(groupe))}
            for code, categorie in enumerate(CATEGORIES):
                comptes = np.bincount(groupe.loc[groupe["categorie"] == code, "rang_critique"], minlength=len(ids))
                bas_ic, haut_ic = _ic_grappes(comptes, totaux, tirages)
                bloc[categorie] = {
                    "part_pct": float(100 * comptes.sum() / max(totaux.sum(), 1)),
                    "ic95_bas": bas_ic, "ic95_haut": haut_ic,
                }
            ligne[nom] = bloc
        aspects.append(ligne)

    p_hors = table["proba_hors_echantillon"].to_numpy()
    p_toutes = toutes["proba_hors_echantillon"].to_numpy()
    p_gele = toutes["proba_modele_gele"].to_numpy()
    categories_toutes, categories_gele = categoriser(p_toutes, bas, haut), categoriser(p_gele, bas, haut)
    mixtes = {}
    for (b, h), nom in (((bas, haut), "seuils_figes"), (SEUILS_SENSIBILITE, "seuils_stricts")):
        mixtes[nom] = {
            "bas": b, "haut": h,
            "critiques_positives_avec_une_phrase_negative": part_critiques_avec_phrase(
                toutes, "proba_hors_echantillon", 1, "negative", b, h),
            "critiques_negatives_avec_une_phrase_positive": part_critiques_avec_phrase(
                toutes, "proba_hors_echantillon", 0, "positive", b, h),
        }
    resultat = {
        "statut": "exploratoire",
        "avertissement": (
            "Le corpus n'a pas d'étiquette par phrase et le modèle est calibré sur des critiques entières. "
            "Ces parts décrivent les scores du modèle : elles ne mesurent pas le sentiment réel des phrases. "
            "Aucune exactitude au niveau de la phrase n'est publiée."
        ),
        "seuils": {"bas": bas, "haut": haut},
        "critiques": int(toutes["id_brut"].nunique()),
        "phrases_scorees": int(len(toutes)),
        "phrases_modelisees": int(len(table)),
        "part_phrases_hors_aspect": float((table["aspect"] == -1).mean()),
        "repartition_scores_phrases_pct": {
            c: float(100 * (categories_toutes == i).mean()) for i, c in enumerate(CATEGORIES)
        },
        "critiques_mixtes": mixtes,
        "accord_vader": mesurer_accord_vader(p_toutes, toutes["vader"].to_numpy(), bas, haut),
        "sensibilite_modele_gele": {
            "correlation_pearson": float(np.corrcoef(p_toutes, p_gele)[0, 1]),
            "meme_categorie": float((categories_toutes == categories_gele).mean()),
            "note": "Le modèle gelé a appris sur ces critiques : ses scores sont donnés pour contrôle seulement.",
        },
        "aspects": aspects,
        "critiques_positives": int((y_par_critique == 1).sum()),
        "critiques_negatives": int((y_par_critique == 0).sum()),
    }
    bm.ecrire_json(CHEMIN_CROISEMENT, resultat)
    return resultat


if __name__ == "__main__":
    import argparse

    analyseur = argparse.ArgumentParser(description="Extension exploratoire au niveau de la phrase.")
    analyseur.add_argument("--etape", choices=["scores", "aspects", "croiser"], required=True)
    etape = analyseur.parse_args().etape
    if etape == "scores":
        print(calculer_scores().shape)
    elif etape == "aspects":
        print(ajuster_aspects().shape)
    else:
        print(json.dumps({k: v for k, v in croiser().items() if k != "aspects"}, ensure_ascii=False, indent=1))

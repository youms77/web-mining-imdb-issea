"""Grille LDA/NMF de la phase 4 : ajustements en cache et cohérence commune.

Aucun ajustement ne lit d'étiquette. Chaque configuration est sauvegardée dès qu'elle est finie :
une exécution interrompue reprend là où elle s'est arrêtée.

Exemple d'utilisation :
    python -m src.topics_grille
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from src import topics as tp
from src.features import CHEMIN_TEXTES
from src.preprocessing import RACINE

DOSSIER = RACINE / "data" / "interim" / "phase4"
DOSSIER_MODELES = DOSSIER / "modeles"
GRILLE_K = (5, 8, 10, 12, 15, 20)  # k = 5 : valeur fixée par le cours, gardée comme repère
FAMILLES = ("lda", "nmf")
N_PROCESSUS = 3


def nom_config(famille: str, k: int, min_df: int = 10, graine: int = tp.GRAINE) -> str:
    """Nom de fichier stable d'une configuration."""
    return f"{famille}_k{k:02d}_mindf{min_df:02d}_g{graine}"


def ecrire_json(chemin, contenu) -> None:
    """Écriture atomique : un fichier interrompu ne passe jamais pour un cache valide."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_name(chemin.name + ".tmp")
    temporaire.write_text(json.dumps(contenu, ensure_ascii=False, indent=1), encoding="utf-8")
    temporaire.replace(chemin)


def ajuster_config(famille: str, k: int, min_df: int = 10, graine: int = tp.GRAINE) -> dict:
    """Ajuste une configuration sur tout le corpus, ou relit son résultat en cache."""
    nom = nom_config(famille, k, min_df, graine)
    chemin_json = DOSSIER_MODELES / f"{nom}.json"
    if chemin_json.exists():
        return json.loads(chemin_json.read_text(encoding="utf-8"))

    if min_df == 10:
        donnees = tp.charger_matrices_topics()
        X, termes = donnees["bow" if famille == "lda" else "tfidf"], donnees["termes"]
    else:
        from src.features import creer_vectoriseur, charger_textes_topics, REGLAGES_TOPICS
        textes = charger_textes_topics()["texte_b"]
        reglages = {**REGLAGES_TOPICS, "min_df": min_df}
        vectoriseur = creer_vectoriseur("bow" if famille == "lda" else "tfidf", **reglages)
        X = vectoriseur.fit_transform(textes)
        termes = vectoriseur.get_feature_names_out().tolist()

    debut = time.perf_counter()
    alertes = 0
    with warnings.catch_warnings(record=True) as captees:
        warnings.simplefilter("always", ConvergenceWarning)
        if famille == "lda":
            modele = tp.ajuster_lda(X, termes, k, graine=graine)
            composantes = modele.get_topics()
            details = {"passes": tp.PASSES_LDA}
        elif famille == "nmf":
            modele = tp.ajuster_nmf(X, k, graine=graine)
            composantes = modele.components_
            details = {"iterations": int(modele.n_iter_),
                       "erreur_reconstruction": float(modele.reconstruction_err_)}
        else:
            raise ValueError("famille doit valoir 'lda' ou 'nmf'.")
        alertes = sum(issubclass(a.category, ConvergenceWarning) for a in captees)
    duree = time.perf_counter() - debut

    DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
    joblib.dump(modele, DOSSIER_MODELES / f"{nom}.joblib", compress=3)
    resultat = {
        "famille": famille, "k": k, "min_df": min_df, "graine": graine,
        "termes_vocabulaire": len(termes), "duree_s": round(duree, 1),
        "alertes_convergence": int(alertes), **details,
        "mots": tp.mots_principaux(composantes, termes),
        "diversite": tp.diversite(tp.mots_principaux(composantes, termes)),
    }
    ecrire_json(chemin_json, resultat)
    print(f"{nom} : {duree:.0f} s", flush=True)
    return resultat


def charger_textes_tokenises() -> list[list[str]]:
    """Textes du pipeline B, dans l'ordre des matrices, sans étiquette."""
    ids = pd.read_parquet(tp.DOSSIER_PHASE3 / "topics_ids.parquet")["id_brut"]
    table = pd.read_parquet(CHEMIN_TEXTES, columns=["id_brut", "texte_b"]).set_index("id_brut")
    return [t.split() for t in table["texte_b"].reindex(ids).to_numpy()]


def calculer_coherences(configs: list[dict], fichier: str = "coherence_grille.json") -> dict:
    """c_v et NPMI de tous les topics en un seul passage sur les textes, pour gagner du temps."""
    chemin = DOSSIER / fichier
    deja = json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else {}
    manquantes = [c for c in configs if nom_config(c["famille"], c["k"], c["min_df"], c["graine"]) not in deja]
    if not manquantes:
        return deja
    topics, reperes = [], []
    for c in manquantes:
        reperes.append((nom_config(c["famille"], c["k"], c["min_df"], c["graine"]), len(topics), len(c["mots"])))
        topics.extend(c["mots"])
    textes = charger_textes_tokenises()
    dictionnaire = Dictionary([sorted({m for t in topics for m in t})])
    mesures = {}
    for mesure in ("c_v", "c_npmi"):
        debut = time.perf_counter()
        mesures[mesure] = tp.coherence_par_topic(topics, textes, dictionnaire, mesure)
        print(f"Cohérence {mesure} : {time.perf_counter() - debut:.0f} s pour {len(topics)} topics", flush=True)
    for nom, depart, n in reperes:
        deja[nom] = {m: mesures[m][depart:depart + n] for m in mesures}
    ecrire_json(chemin, deja)
    return deja


def executer_grille() -> list[dict]:
    """Ajuste les 12 configurations de la grille puis calcule leur cohérence."""
    DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
    debut = time.perf_counter()
    taches = [(f, k) for k in GRILLE_K for f in FAMILLES]
    configs = Parallel(n_jobs=N_PROCESSUS, backend="loky")(
        delayed(ajuster_config)(f, k) for f, k in taches
    )
    calculer_coherences(configs)
    print(f"Grille terminée en {(time.perf_counter() - debut) / 60:.1f} min", flush=True)
    return configs


if __name__ == "__main__":
    executer_grille()

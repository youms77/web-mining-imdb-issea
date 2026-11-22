"""Phase 7 : prépare `models/deploy/` et ajoute la clé `application` à `metrics.json`.

Ce script se lance une fois, en local, car il lit les caches de `data/interim/`. L'application
déployée n'en dépend pas. Il refuse d'écrire si une clé existante de `metrics.json` a changé.
Aucune étiquette ni aucun texte du test n'est lu pour fixer les seuils ou choisir un exemple.

Exemple d'utilisation :
    .\\.venv\\Scripts\\python.exe -m src.app_artefacts
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from src import application as ap
from src.data import charger_brut
from src.features import CHEMIN_TEXTES, creer_vectoriseur_topics
from src.models import preparer_texte
from src.preprocessing import DOSSIER_NLTK, RACINE, charger_ressources_b, pipeline_b

GRAINE = 42
DEPLOIEMENT = ap.DOSSIER_DEPLOIEMENT
PHASE3 = RACINE / "data" / "interim" / "phase3"
PHASE4 = RACINE / "data" / "interim" / "phase4"
CHEMIN_OOF = RACINE / "data" / "interim" / "phase5" / "probabilites_hors_pli.parquet"
SHA_WORDNET = "CBDA5EA6EEF7F36A97A43D4A75F85E07FCCBB4F23657D27B4CCBC93E2646AB59"
SHA_ETIQUETEUR = "6025F530624335C67D6547D44757B357B4E79BAE030A0383E9887A92C1718F0B"

PERCENTILE_MOTS_FIABLES = 1.0   # moins de 1 % des critiques d'entraînement sont plus courtes
PERCENTILE_LANGUE = 0.1         # seuils de langue placés très bas pour éviter les faux refus
PART_LETTRES_LATINES_MIN = 0.5  # règle simple : une majorité de lettres latines
TRANCHES_LONGUEUR = [(0, 20), (20, 30), (30, 40), (40, 60), (60, 100), (100, 200), (200, None)]
LONGUEUR_EXTRAIT = 160
N_CONTROLE_B = 300
ID_EXEMPLE_MITIGE = 13669  # relue : avis partagé, sans nom de personne ni donnée personnelle
# Choix après lecture : sans nom d'acteur réel ni détail sur l'auteur de la critique.
EXTRAITS_RETENUS = {0: 27646, 1: 39679, 2: 7586, 3: 4201, 4: 46199, 5: 21309, 6: 12719, 7: 16137}

EXEMPLE_POSITIF = (
    "This film is a quiet triumph. The performances are warm and honest, the photography is beautiful, "
    "and the script trusts its audience from the first scene to the last. I was moved by the relationship "
    "between the two brothers, and the ending is perfect. I loved every minute and I would happily watch "
    "it again with the whole family."
)
EXEMPLE_NEGATIF = (
    "This movie is a complete waste of time. The plot makes no sense, the dialogue is painfully stupid, "
    "and the acting is wooden from start to finish. The jokes fall flat, the editing is a mess, and the "
    "ending is even worse than the rest. I was bored after ten minutes and I regret paying for a ticket. "
    "Avoid it."
)


def _sha256(chemin) -> str:
    return hashlib.sha256(chemin.read_bytes()).hexdigest().upper()


def copier_modele_svm(metriques: dict) -> dict:
    """Copie le SVM gelé sans le réécrire, puis contrôle son empreinte."""
    source = RACINE / "models" / ap.FICHIER_SVM
    attendu = metriques["sentiment"]["choix_figes"]["modeles"]["svm_lineaire"]["sha256"]
    if _sha256(source) != attendu:
        raise RuntimeError("Le modèle gelé ne correspond plus à son empreinte.")
    cible = DEPLOIEMENT / ap.FICHIER_SVM
    shutil.copyfile(source, cible)
    if _sha256(cible) != attendu:
        raise RuntimeError("Copie du modèle gelé incorrecte.")
    return {"sha256_identique_au_modele_gele": True, "taille_mo": round(cible.stat().st_size / 1024 ** 2, 2)}


def construire_topics() -> dict:
    """Réajuste l'IDF du vectoriseur de topics et vérifie qu'il redonne la matrice de la phase 3."""
    ids = pd.read_parquet(PHASE3 / "topics_ids.parquet")["id_brut"].to_numpy()
    textes = pd.read_parquet(CHEMIN_TEXTES, columns=["id_brut", "texte_b"]).set_index("id_brut")["texte_b"]
    vectoriseur = creer_vectoriseur_topics("nmf").fit(textes.reindex(ids))
    termes = json.loads((PHASE3 / "topics_vocabulaire.json").read_text(encoding="utf-8"))["termes"]
    if vectoriseur.get_feature_names_out().tolist() != termes:
        raise RuntimeError("Le vocabulaire des topics diffère de la phase 3.")

    reference = sparse.load_npz(PHASE3 / "topics_tfidf.npz").tocsr()
    indices = np.random.default_rng(GRAINE).choice(len(ids), 2000, replace=False)
    recalcul = vectoriseur.transform(textes.reindex(ids[indices]))
    ecart = abs(recalcul - reference[indices])
    ecart_max = float(ecart.max()) if ecart.nnz else 0.0
    if ecart_max > 1e-6:
        raise RuntimeError(f"TF-IDF différent de la phase 3 : écart {ecart_max}.")

    nmf = joblib.load(PHASE4 / "modeles" / "nmf_k08_mindf10_g42.joblib")
    cible = DEPLOIEMENT / ap.FICHIER_TOPICS
    joblib.dump({"vectoriseur": vectoriseur, "nmf": nmf}, cible, compress=("xz", 3))

    theta = np.load(PHASE4 / "theta_nmf_k08_mindf10_g42.npy")[indices[:300]]
    poids = nmf.transform(reference[indices[:300]])
    sommes = poids.sum(axis=1, keepdims=True)
    parts = np.divide(poids, sommes, out=np.zeros_like(poids), where=sommes > 0)
    utiles = sommes.ravel() > 0
    return {
        "taille_mo": round(cible.stat().st_size / 1024 ** 2, 2),
        "tfidf_ecart_max_sur_2000_critiques": ecart_max,
        "nmf_meme_theme_principal_sur_300_critiques": float((parts[utiles].argmax(1) == theta[utiles].argmax(1)).mean()),
        "nmf_ecart_absolu_moyen_des_parts": float(np.abs(parts[utiles] - theta[utiles]).mean()),
    }


def copier_nltk() -> dict:
    """Copie WordNet et l'étiqueteur, avec leurs licences, sans les mots vides NLTK."""
    corpora, taggers = DEPLOIEMENT / "nltk_data" / "corpora", DEPLOIEMENT / "nltk_data" / "taggers"
    corpora.mkdir(parents=True, exist_ok=True)
    taggers.mkdir(parents=True, exist_ok=True)
    wordnet = corpora / "wordnet.zip"
    etiqueteur = taggers / "averaged_perceptron_tagger_eng.zip"
    shutil.copyfile(DOSSIER_NLTK / "corpora" / "wordnet.zip", wordnet)
    shutil.copyfile(DOSSIER_NLTK / "taggers" / "averaged_perceptron_tagger_eng.zip", etiqueteur)
    if _sha256(wordnet) != SHA_WORDNET or _sha256(etiqueteur) != SHA_ETIQUETEUR:
        raise RuntimeError("Empreinte NLTK inattendue.")
    with zipfile.ZipFile(wordnet) as archive:
        licence = archive.read("wordnet/LICENSE").decode("utf-8", errors="replace")
    (DEPLOIEMENT / "nltk_data" / "LICENCES.md").write_text(
        "# Licences des ressources linguistiques embarquées\n\n"
        "## Étiqueteur grammatical (averaged_perceptron_tagger_eng)\n"
        "Licence MIT indiquée par l'index officiel `nltk_data`. Code d'origine : Copyright 2013 Matthew "
        "Honnibal ; modifications NLTK : Copyright 2015 The NLTK Project (en-tête de `nltk/tag/perceptron.py`). "
        "Corpus d'entraînement des poids : [à vérifier].\n\n"
        "## WordNet 3.0 (Princeton University)\nNotice reproduite telle quelle, comme la licence l'exige :\n\n"
        f"```\n{licence.strip()}\n```\n",
        encoding="utf-8",
    )
    return {
        "wordnet_mo": round(wordnet.stat().st_size / 1024 ** 2, 2),
        "etiqueteur_mo": round(etiqueteur.stat().st_size / 1024 ** 2, 2),
    }


def _couper(extrait: str, taille: int = LONGUEUR_EXTRAIT) -> str:
    """Coupe un extrait à la fin d'un mot, sous `taille` caractères."""
    propre = extrait.strip().removeprefix("…").removesuffix("…").strip()
    if len(propre) <= taille:
        return f"… {propre} …"
    return "… " + propre[:taille].rsplit(" ", 1)[0] + " …"


def choisir_extraits(id_train: set[int], metriques: dict) -> list[dict]:
    """Un court extrait par thème NMF, choisi après relecture parmi les extraits de la phase 4."""
    brut = json.loads((PHASE4 / "extraits_topics.json").read_text(encoding="utf-8"))["nmf"]
    sortie = []
    for fiche in metriques["topics"]["fiches_topics"]["nmf"]:
        topic, identifiant = fiche["topic"], EXTRAITS_RETENUS[fiche["topic"]]
        retenu = next(c for c in brut[topic] if c["id_brut"] == identifiant)
        if identifiant not in id_train:
            raise RuntimeError("Un extrait affiché doit venir de l'entraînement.")
        sortie.append({"topic": topic, "nom": fiche["nom"], "id_brut": identifiant, "extrait": _couper(retenu["extrait"])})
    return sortie


def ecrire_liste_extraits(contenu: dict) -> None:
    """Liste tout texte du corpus embarqué, pour la décision de publication de la phase 8."""
    mitige = contenu["exemples"]["mitige"]
    lignes = [
        "# Textes du corpus affichés par l'application",
        "",
        "Fichier produit par `src/app_artefacts.py`. **Licence du corpus : [à vérifier].** "
        "Décision de publication à prendre en phase 8, avant tout dépôt public.",
        "",
        "Relecture faite en phase 7 : pas d'adresse, de courriel, de lien ni d'information sur la vie privée "
        "d'un auteur. Les extraits sont en minuscules, car ils viennent du texte nettoyé.",
        "",
        "## Critique entière (exemple « mitigé »)",
        "",
        f"- `id_brut` {mitige['id_brut']}, partie entraînement, {len(mitige['texte'].split())} mots.",
        f"- Texte : {mitige['texte']}",
        "",
        f"## Extraits de thèmes NMF ({len(contenu['extraits'])}, {LONGUEUR_EXTRAIT} caractères au plus hors points de suspension)",
        "",
        "| Thème | `id_brut` | Extrait |",
        "|---|---|---|",
    ]
    lignes += [f"| {e['nom']} | {e['id_brut']} | {e['extrait']} |" for e in contenu["extraits"]]
    lignes += [
        "",
        "## Si la publication de ces textes est refusée",
        "",
        "Vider `extraits` dans `models/deploy/contenu_app.json` et remplacer l'exemple mitigé par un texte rédigé. "
        "L'application masque alors les extraits ; le reste fonctionne sans eux.",
        "",
    ]
    (RACINE / "docs" / "extraits_application.md").write_text("\n".join(lignes), encoding="utf-8")


def mesurer_seuils(train: pd.DataFrame, artefacts: ap.Artefacts, metriques: dict) -> dict:
    """Seuils de longueur et de langue, mesurés sur l'entraînement seul."""
    jetons = train["texte"].str.split()
    longueurs = jetons.str.len().to_numpy()
    signaux = np.array([ap.signaux_langue(j, artefacts.mots_vides, artefacts.unigrammes) for j in jetons])
    seuil_vides = float(np.percentile(signaux[:, 0], PERCENTILE_LANGUE))
    seuil_connus = float(np.percentile(signaux[:, 1], PERCENTILE_LANGUE))
    refus = (signaux[:, 0] < seuil_vides) & (signaux[:, 1] < seuil_connus)
    longueurs_corpus = metriques["donnees"]["longueurs"]
    return {
        "mots_min_analyse": int(longueurs.min()),
        "mots_min_fiable": int(np.percentile(longueurs, PERCENTILE_MOTS_FIABLES)),
        "percentile_mots_min_fiable": PERCENTILE_MOTS_FIABLES,
        "part_entrainement_sous_le_seuil_fiable": float((longueurs < np.percentile(longueurs, PERCENTILE_MOTS_FIABLES)).mean()),
        "caracteres_max": int(max(longueurs_corpus[c]["caracteres"]["maximum"] for c in ("negative", "positive"))),
        "part_lettres_latines_min": PART_LETTRES_LATINES_MIN,
        "part_mots_vides_min": seuil_vides,
        "part_mots_connus_min": seuil_connus,
        "percentile_seuils_langue": PERCENTILE_LANGUE,
        "critiques_entrainement_refusees_par_la_langue": int(refus.sum()),
        "part_entrainement_refusee_par_la_langue": float(refus.mean()),
    }


def fiabilite_par_longueur(train: pd.DataFrame) -> list[dict]:
    """Scores hors pli du SVM calibré, par longueur. Ni le test ni le modèle final n'interviennent."""
    longueurs = train["texte"].str.split().str.len().to_numpy()
    y, p = train["y"].to_numpy(), train["p_oof"].to_numpy()
    lignes = []
    for bas, haut in TRANCHES_LONGUEUR:
        masque = (longueurs >= bas) & (longueurs < (haut if haut is not None else np.inf))
        lignes.append({
            "mots_min": bas, "mots_max_exclu": haut, "effectif": int(masque.sum()),
            "part": float(masque.mean()),
            "exactitude": float(((p[masque] >= 0.5) == y[masque]).mean()),
            "confiance_moyenne": float(np.maximum(p[masque], 1 - p[masque]).mean()),
            "brier": float(np.mean((p[masque] - y[masque]) ** 2)),
        })
    return lignes


def controler_pipeline_b(brut: pd.DataFrame) -> dict:
    """Compare le pipeline B de l'application au cache de la phase 2, sur un échantillon."""
    cache = pd.read_parquet(CHEMIN_TEXTES, columns=["id_brut", "texte_b"]).sample(N_CONTROLE_B, random_state=GRAINE)
    textes = brut.set_index("id_brut")["review"].reindex(cache["id_brut"])
    identiques = sum(pipeline_b(t) == b for t, b in zip(textes, cache["texte_b"]))
    return {"critiques": N_CONTROLE_B, "identiques": int(identiques)}


def ecrire_metriques(metriques: dict, application: dict) -> None:
    """Ajoute la seule clé `application`, après contrôle des autres clés."""
    chemin = ap.CHEMIN_METRIQUES
    actuel = json.loads(chemin.read_text(encoding="utf-8"))
    for cle, valeur in actuel.items():
        if cle != "application" and valeur != metriques[cle]:
            raise RuntimeError(f"La clé `{cle}` de metrics.json a changé : écriture refusée.")
    actuel["application"] = {**actuel.get("application", {}), **application}
    chemin.write_text(json.dumps(actuel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    DEPLOIEMENT.mkdir(parents=True, exist_ok=True)
    metriques = ap.charger_metriques()
    brut = charger_brut()
    oof = pd.read_parquet(CHEMIN_OOF, columns=["id_brut", "y", "svm_lineaire__sigmoid"]).rename(
        columns={"svm_lineaire__sigmoid": "p_oof"}
    )
    textes = pd.read_parquet(RACINE / "data" / "interim" / "phase5" / "textes_sans_notes.parquet").set_index("id_brut")["texte"]
    train = oof.assign(texte=textes.reindex(oof["id_brut"]).to_numpy())
    id_train = set(train["id_brut"].tolist())
    if ID_EXEMPLE_MITIGE not in id_train:
        raise RuntimeError("L'exemple mitigé doit venir de l'entraînement.")

    controle_svm = copier_modele_svm(metriques)
    controle_topics = construire_topics()
    controle_nltk = copier_nltk()
    critique = brut.loc[brut["id_brut"] == ID_EXEMPLE_MITIGE, "review"].iloc[0]
    contenu = {
        "exemples": {
            "positif": {"texte": EXEMPLE_POSITIF, "origine": "Texte rédigé pour la démonstration."},
            "negatif": {"texte": EXEMPLE_NEGATIF, "origine": "Texte rédigé pour la démonstration."},
            "mitige": {"texte": critique, "origine": "Vraie critique du corpus, partie entraînement.", "id_brut": ID_EXEMPLE_MITIGE},
        },
        "extraits": choisir_extraits(id_train, metriques),
    }
    (DEPLOIEMENT / ap.FICHIER_CONTENU).write_text(json.dumps(contenu, ensure_ascii=False, indent=2), encoding="utf-8")
    ecrire_liste_extraits(contenu)

    # Les seuils n'existent pas encore : seuls les vocabulaires sont nécessaires pour les mesurer.
    ressources = charger_ressources_b()
    modele = joblib.load(DEPLOIEMENT / ap.FICHIER_SVM)
    vocabulaire = modele.calibrated_classifiers_[0].estimator.named_steps["vectoriseur"].vocabulary_
    outils = ap.Artefacts(
        modele=modele, vectoriseur=None, coefficients=np.empty(0), constante=0.0, vectoriseur_topics=None, nmf=None,
        mots_vides=frozenset(ressources["mots_vides"]), unigrammes=frozenset(t for t in vocabulaire if " " not in t),
        metriques=metriques, contenu=contenu, fiches_topics=[],
    )
    seuils = mesurer_seuils(train, outils, metriques)
    ligne = train.loc[train["id_brut"] == ID_EXEMPLE_MITIGE].iloc[0]
    application = {
        "statut": "seuils et contrôles de la phase 7, mesurés sur l'entraînement seul",
        "seuils_texte": seuils,
        "regle_seuil_texte_court": (
            "Moins de mots utiles que le 1er percentile de l'entraînement : score affiché comme non fiable. "
            "Sous la plus courte critique d'entraînement : aucun score."
        ),
        "fiabilite_par_longueur_hors_pli": fiabilite_par_longueur(train),
        "exemple_mitige": {
            "id_brut": ID_EXEMPLE_MITIGE,
            "classe_reelle": "positif" if int(ligne["y"]) == 1 else "négatif",
            "probabilite_hors_pli": float(ligne["p_oof"]),
            "probabilite_modele_gele": float(modele.predict_proba([preparer_texte(critique)])[0, 1]),
            "note": "Critique d'entraînement : le modèle gelé l'a vue, son score est donc plus tranché que le score hors pli.",
        },
        "controles_artefacts": {
            "svm": controle_svm, "topics": controle_topics, "nltk": controle_nltk,
            "pipeline_b": controler_pipeline_b(brut),
        },
    }
    ecrire_metriques(metriques, application)
    print(json.dumps(application, ensure_ascii=False, indent=1))
    print(json.dumps(contenu["extraits"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

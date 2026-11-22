"""Mesures de la phase 3 : vocabulaire, creux, effet des réglages, BoW contre TF-IDF.

Toute comparaison liée à un score se fait en validation croisée sur l'entraînement.
Aucune étiquette du test n'est lue ici, sauf par la stratification de `separer_entrainement_test`,
qui reproduit la séparation déjà définie. L'application n'a pas besoin de ce module.

Exemple d'utilisation (reprend les caches existants) :
    python -m src.features_eval
Pour tout recalculer :
    python -m src.features_eval --recalculer
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import time
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import exploration as ex
from src import features as f
from src.data import charger_brut, cle_sans_ponctuation, dedoublonner, separer_entrainement_test
from src.preprocessing import RACINE

GRAINE = 42
N_PLIS = 5
N_PROCESSUS = 3
N_PROCESSUS_LOURD = 2  # trigrammes et min_df=1 : vocabulaire très grand, mémoire limitée
DOSSIER_CACHE = RACINE / "data" / "interim" / "phase3"

# Règles fixées avant de lire les résultats.
SEUIL_GAIN_MINIMAL = 0.0005  # 0,05 point d'exactitude, même règle qu'en phase 2
SEUIL_OCCURRENCES_TOPICS = 95.0  # part minimale des occurrences gardée par min_df (topics)
MAX_DF_TOPICS = 0.5  # valeur du cours (`no_above=0.5`)
TAILLE_FRAGMENT = 10  # fragments de 10 mots pour repérer les quasi-doublons
SEUILS_QUASI_DOUBLONS = (0.5, 0.8, 0.9)

GRILLE_MIN_DF = (1, 2, 5, 10, 20, 50)
GRILLE_MAX_DF = (1.0, 0.9, 0.7, 0.5, 0.3)
GRILLE_NGRAM = ((1, 1), (1, 2), (1, 3))

REFERENCE = {"ngram_range": (1, 2), "min_df": 5, "max_df": 1.0, "sublinear_tf": True}
VARIANTES_TFIDF = {
    "reference": {},
    "min_df_1": {"min_df": 1},
    "min_df_2": {"min_df": 2},
    "min_df_10": {"min_df": 10},
    "min_df_20": {"min_df": 20},
    "min_df_50": {"min_df": 50},
    "max_df_0.9": {"max_df": 0.9},
    "max_df_0.7": {"max_df": 0.7},
    "max_df_0.5": {"max_df": 0.5},
    "max_df_0.3": {"max_df": 0.3},
    "unigrammes": {"ngram_range": (1, 1)},
    "trigrammes": {"ngram_range": (1, 3)},
    "tf_brut": {"sublinear_tf": False},
    # Contrôle du mécanisme : sans bigrammes, « not » disparaît vraiment avec max_df = 0,7.
    "unigrammes_max_df_0.7": {"ngram_range": (1, 1), "max_df": 0.7},
}
VARIANTES_HORS_FIGURE = ("reference", "unigrammes_max_df_0.7")
LIBELLES_VARIANTES = {
    "min_df_1": "min_df = 1", "min_df_2": "min_df = 2", "min_df_10": "min_df = 10",
    "min_df_20": "min_df = 20", "min_df_50": "min_df = 50", "max_df_0.9": "max_df = 0,9",
    "max_df_0.7": "max_df = 0,7", "max_df_0.5": "max_df = 0,5", "max_df_0.3": "max_df = 0,3",
    "unigrammes": "mots seuls (1,1)", "trigrammes": "jusqu'aux trigrammes (1,3)",
    "tf_brut": "sublinear_tf = False",
}
MODELES = {
    "naive_bayes_bow": ("naive_bayes", "bow"),
    "naive_bayes_tfidf": ("naive_bayes", "tfidf"),
    "regression_logistique_bow": ("regression_logistique", "bow"),
    "svm_lineaire_bow": ("svm_lineaire", "bow"),
    "svm_lineaire_tfidf": ("svm_lineaire", "tfidf"),
}
LIBELLES_MODELES = {
    "naive_bayes": "Naive Bayes", "regression_logistique": "Régression logistique",
    "svm_lineaire": "SVM linéaire",
}
COULEUR_BOW, COULEUR_TFIDF = "#eda100", "#6250d6"
RAMPE_NGRAM = {"1_1": "#a79df0", "1_2": "#6250d6", "1_3": "#2f2580"}
LIBELLES_NGRAM = {"1_1": "mots seuls", "1_2": "+ bigrammes", "1_3": "+ trigrammes"}


# --- Caches -----------------------------------------------------------------------------------

def _lire_json(nom: str) -> dict | None:
    chemin = DOSSIER_CACHE / nom
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else None


def _ecrire_json(nom: str, contenu: dict) -> None:
    DOSSIER_CACHE.mkdir(parents=True, exist_ok=True)
    temporaire = DOSSIER_CACHE / f"{nom}.tmp"
    temporaire.write_text(json.dumps(contenu, ensure_ascii=False, indent=1), encoding="utf-8")
    temporaire.replace(DOSSIER_CACHE / nom)


# --- Séparation et textes communs ---------------------------------------------------------------

def fragments(texte: str, taille: int = TAILLE_FRAGMENT) -> set[str]:
    """Ensemble des suites de `taille` mots d'un texte (le texte entier s'il est plus court)."""
    mots = texte.split()
    if len(mots) < taille:
        return {" ".join(mots)}
    return {" ".join(mots[i:i + taille]) for i in range(len(mots) - taille + 1)}


def mesurer_quasi_doublons(
    textes_train: list[str], textes_test: list[str], taille: int = TAILLE_FRAGMENT
) -> pd.DataFrame:
    """Paires entraînement/test partageant des fragments, avec contenance et Jaccard.

    Contenance : fragments communs / fragments du plus court des deux textes.
    """
    index: dict[str, list[int]] = collections.defaultdict(list)
    tailles_test = []
    for j, texte in enumerate(textes_test):
        ensemble = fragments(texte, taille)
        tailles_test.append(len(ensemble))
        for fragment in ensemble:
            index[fragment].append(j)
    lignes = []
    for i, texte in enumerate(textes_train):
        ensemble = fragments(texte, taille)
        compte: collections.Counter = collections.Counter()
        for fragment in ensemble:
            if fragment in index:
                compte.update(index[fragment])
        for j, communs in compte.items():
            a, b = len(ensemble), tailles_test[j]
            lignes.append((i, j, communs, communs / min(a, b), communs / (a + b - communs)))
    return pd.DataFrame(lignes, columns=["i_train", "j_test", "fragments_communs", "contenance", "jaccard"])


def controler_separation(recalculer: bool = False) -> dict:
    """Recalcule la séparation de src/data.py, prouve l'identité au cache et sauvegarde les index."""
    corpus = dedoublonner(charger_brut(), mode="sans_ponctuation")
    train, test = separer_entrainement_test(corpus, graine=GRAINE)
    id_train, id_test = train["id_brut"].to_numpy(), test["id_brut"].to_numpy()

    textes = pd.read_parquet(f.CHEMIN_TEXTES, columns=["id_brut", "partition", "texte_a", "texte_b"])
    cache_train = textes.loc[textes["partition"] == f.PARTITION_TRAIN, "id_brut"].to_numpy()
    cache_test = textes.loc[textes["partition"] == f.PARTITION_TEST, "id_brut"].to_numpy()
    identique_cache = bool(
        np.array_equal(np.sort(id_train), np.sort(cache_train))
        and np.array_equal(np.sort(id_test), np.sort(cache_test))
    )
    if not identique_cache:
        raise ValueError("La séparation recalculée diffère des textes en cache.")

    if f.CHEMIN_INDEX.exists() and not recalculer:
        sauve_train, sauve_test = f.charger_index()
        if not (np.array_equal(np.sort(sauve_train), np.sort(id_train))
                and np.array_equal(np.sort(sauve_test), np.sort(id_test))):
            raise ValueError("Les index sauvegardés diffèrent de la séparation de src/data.py.")
    else:
        f.sauvegarder_index(id_train, id_test)
    relu_train, relu_test = f.charger_index()

    communs = lambda a, b: len(set(a) & set(b))
    masque = textes["partition"] == f.PARTITION_TRAIN
    a_train, a_test = textes.loc[masque, "texte_a"].tolist(), textes.loc[~masque, "texte_a"].tolist()
    b_train, b_test = textes.loc[masque, "texte_b"], textes.loc[~masque, "texte_b"]

    chemin_paires = DOSSIER_CACHE / "quasi_doublons_train_test.parquet"
    if chemin_paires.exists() and not recalculer:
        paires = pd.read_parquet(chemin_paires)
    else:
        brut = mesurer_quasi_doublons(a_train, a_test)
        ids_a = textes.loc[masque, "id_brut"].to_numpy()
        ids_b = textes.loc[~masque, "id_brut"].to_numpy()
        paires = brut[brut["contenance"] >= min(SEUILS_QUASI_DOUBLONS)].copy()
        paires.insert(0, "id_brut_train", ids_a[paires["i_train"].to_numpy()])
        paires.insert(1, "id_brut_test", ids_b[paires["j_test"].to_numpy()])
        paires = paires.drop(columns=["i_train", "j_test"]).sort_values("id_brut_test").reset_index(drop=True)
        DOSSIER_CACHE.mkdir(parents=True, exist_ok=True)
        paires.to_parquet(chemin_paires, index=False)

    quasi = {}
    for seuil in SEUILS_QUASI_DOUBLONS:
        retenues = paires[paires["contenance"] >= seuil]
        quasi[f"contenance_{seuil}"] = {
            "paires": int(len(retenues)),
            "critiques_de_test": int(retenues["id_brut_test"].nunique()),
            "part_du_test_pct": round(100 * retenues["id_brut_test"].nunique() / len(id_test), 3),
        }
    return {
        "methode": "80/20 stratifié, graine 42, après dédoublonnage (src/data.py)",
        "entrainement": int(len(id_train)),
        "test": int(len(id_test)),
        "fichier_index": "data/interim/separation_80_20.parquet",
        "empreinte_index_entrainement": f.empreinte_index(id_train),
        "empreinte_index_test": f.empreinte_index(id_test),
        "identique_aux_textes_en_cache": identique_cache,
        "identique_apres_relecture": bool(
            np.array_equal(np.sort(relu_train), np.sort(id_train))
            and np.array_equal(np.sort(relu_test), np.sort(id_test))
        ),
        "index_communs": int(np.intersect1d(id_train, id_test).size),
        "textes_communs": {
            "brut": communs(train["review"], test["review"]),
            "cle_sans_ponctuation": communs(
                train["review"].map(cle_sans_ponctuation), test["review"].map(cle_sans_ponctuation)
            ),
            "pipeline_a": communs(a_train, a_test),
            "pipeline_b": communs(b_train, b_test),
        },
        "quasi_doublons_entre_parties": {
            "definition": (
                f"fragments de {TAILLE_FRAGMENT} mots du pipeline A ; contenance = fragments "
                "communs / fragments du plus court texte. Étiquettes non lues."
            ),
            "seuils": quasi,
            "fichier": "data/interim/phase3/quasi_doublons_train_test.parquet",
        },
    }


# --- Vocabulaire et creux -----------------------------------------------------------------------

def _masque_termes(df: np.ndarray, n_docs: int, min_df: int, max_df: float) -> np.ndarray:
    """Même règle que scikit-learn : min_df <= df <= max_df x nombre de documents."""
    return (df >= min_df) & (df <= max_df * n_docs)


def _ligne_vocabulaire(df, occurrences, n_docs, min_df, max_df, indice_not=None) -> dict:
    garde = _masque_termes(df, n_docs, min_df, max_df)
    termes = int(garde.sum())
    non_nuls = int(df[garde].sum())
    ligne = {
        "min_df": min_df,
        "max_df": max_df,
        "termes": termes,
        "taux_de_creux_pct": round(100 * (1 - non_nuls / (n_docs * termes)), 4) if termes else 100.0,
        "termes_non_nuls_par_document": round(non_nuls / n_docs, 1),
        "occurrences_conservees_pct": round(100 * float(occurrences[garde].sum() / occurrences.sum()), 2),
    }
    if indice_not is not None:
        ligne["not_conserve"] = bool(garde[indice_not])
    return ligne


def mesurer_vocabulaire_a(textes: np.ndarray) -> dict:
    """Effet de ngram_range, min_df et max_df sur le pipeline A, sur l'entraînement seul."""
    resultats = {}
    n_docs = len(textes)
    for ngram in GRILLE_NGRAM:
        debut = time.perf_counter()
        vectoriseur = f.creer_bow(ngram_range=ngram)
        matrice = vectoriseur.fit_transform(textes).tocsc()
        df = np.diff(matrice.indptr)
        occurrences = np.asarray(matrice.sum(axis=0)).ravel()
        indice_not = vectoriseur.vocabulary_["not"]
        resultats[f"{ngram[0]}_{ngram[1]}"] = {
            "effet_min_df": [
                _ligne_vocabulaire(df, occurrences, n_docs, v, 1.0, indice_not) for v in GRILLE_MIN_DF
            ],
            "effet_max_df": [
                _ligne_vocabulaire(df, occurrences, n_docs, 5, v, indice_not) for v in GRILLE_MAX_DF
            ],
            "duree_s": round(time.perf_counter() - debut, 1),
        }
        del vectoriseur, matrice
    return resultats


def mesurer_matrice_reference(textes: np.ndarray) -> dict:
    """Profil de la matrice de référence et effet de sublinear_tf sur les poids."""
    bow = f.creer_bow(ngram_range=REFERENCE["ngram_range"], min_df=REFERENCE["min_df"]).fit_transform(textes)
    profil = f.profiler_matrice(bow)
    effet = {}
    for sublineaire in (False, True):
        matrice = f.creer_tfidf(
            ngram_range=REFERENCE["ngram_range"], min_df=REFERENCE["min_df"], sublinear_tf=sublineaire
        ).fit_transform(textes)
        poids_max = matrice.max(axis=1).toarray().ravel()
        effet[f"sublinear_{sublineaire}"] = {"poids_maximal_moyen_par_document": round(float(poids_max.mean()), 4)}
    return {"profil_bow_et_tfidf": profil, "effet_sublinear_tf": effet}


# --- Validation croisée -------------------------------------------------------------------------

def _config(nom: str) -> tuple[str, str, dict]:
    if nom in VARIANTES_TFIDF:
        return "regression_logistique", "tfidf", {**REFERENCE, **VARIANTES_TFIDF[nom]}
    modele, nature = MODELES[nom]
    reglages = {k: v for k, v in REFERENCE.items() if nature == "tfidf" or k != "sublinear_tf"}
    return modele, nature, reglages


def _evaluer_pli(pipeline, textes, y, indices_fit, indices_val) -> dict:
    """Le vectoriseur et le classifieur ne voient que la partie d'ajustement du pli."""
    modele = clone(pipeline)
    with warnings.catch_warnings(record=True) as alertes:
        warnings.simplefilter("always", ConvergenceWarning)
        modele.fit(textes[indices_fit], y[indices_fit])
    predictions = modele.predict(textes[indices_val])
    if hasattr(modele, "decision_function"):
        score = modele.decision_function(textes[indices_val])
    else:
        score = modele.predict_proba(textes[indices_val])[:, 1]
    return {
        "exactitude": float(accuracy_score(y[indices_val], predictions)),
        "f1": float(f1_score(y[indices_val], predictions)),
        "roc_auc": float(roc_auc_score(y[indices_val], score)),
        "termes": len(modele.named_steps["vectoriseur"].vocabulary_),
        "alertes_convergence": sum(issubclass(a.category, ConvergenceWarning) for a in alertes),
    }


def evaluer_config(nom: str, textes: np.ndarray, y: np.ndarray) -> dict:
    """Validation croisée stratifiée à 5 plis, avec les mêmes plis qu'en phase 2."""
    modele, nature, reglages = _config(nom)
    pipeline = f.creer_pipeline_supervise(modele, nature, **reglages)
    plis = list(StratifiedKFold(n_splits=N_PLIS, shuffle=True, random_state=GRAINE).split(textes, y))
    lourd = reglages["ngram_range"] == (1, 3) or reglages["min_df"] == 1
    debut = time.perf_counter()
    resultats = Parallel(n_jobs=N_PROCESSUS_LOURD if lourd else N_PROCESSUS)(
        delayed(_evaluer_pli)(pipeline, textes, y, a, v) for a, v in plis
    )
    return {
        "modele": modele,
        "vectoriseur": nature,
        "reglages": {k: list(v) if isinstance(v, tuple) else v for k, v in reglages.items()},
        "exactitude_par_pli": [round(r["exactitude"], 5) for r in resultats],
        "exactitude": round(float(np.mean([r["exactitude"] for r in resultats])), 5),
        "f1": round(float(np.mean([r["f1"] for r in resultats])), 5),
        "roc_auc": round(float(np.mean([r["roc_auc"] for r in resultats])), 5),
        "termes_moyens": int(np.mean([r["termes"] for r in resultats])),
        "alertes_convergence": int(sum(r["alertes_convergence"] for r in resultats)),
        "duree_s": round(time.perf_counter() - debut, 1),
    }


def decider(reference: dict, variante: dict) -> dict:
    """Écart apparié par pli. Une variante gagne si l'écart dépasse 0,05 point et 2 erreurs-types."""
    ecarts = np.array(variante["exactitude_par_pli"]) - np.array(reference["exactitude_par_pli"])
    gain = float(ecarts.mean())
    erreur_type = float(ecarts.std(ddof=1) / math.sqrt(len(ecarts)))
    return {
        "gain_exactitude_points": round(100 * gain, 3),
        "erreur_type_points": round(100 * erreur_type, 3),
        "plis_gagnants": int((ecarts > 0).sum()),
        "ameliore": bool(gain >= SEUIL_GAIN_MINIMAL and gain > 2 * erreur_type),
        "degrade": bool(gain <= -SEUIL_GAIN_MINIMAL and -gain > 2 * erreur_type),
    }


def chronometrer(textes: np.ndarray, y: np.ndarray) -> dict:
    """Mesure chaque ajustement sur 2 000 critiques, puis extrapole la validation croisée."""
    tirage = np.random.default_rng(GRAINE)
    indices = np.concatenate([
        tirage.choice(np.flatnonzero(y == classe), size=1000, replace=False) for classe in (0, 1)
    ])
    taille_pli = len(textes) * (N_PLIS - 1) / N_PLIS
    total, details = 0.0, {}
    for nom in [*VARIANTES_TFIDF, *MODELES]:
        modele, nature, reglages = _config(nom)
        debut = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            f.creer_pipeline_supervise(modele, nature, **{**reglages, "min_df": min(reglages["min_df"], 2)}).fit(
                textes[indices], y[indices]
            )
        duree = time.perf_counter() - debut
        lourd = reglages["ngram_range"] == (1, 3) or reglages["min_df"] == 1
        minutes = duree * taille_pli / 2000 * N_PLIS / (N_PROCESSUS_LOURD if lourd else N_PROCESSUS) / 60
        details[nom] = round(duree, 2)
        total += minutes
    return {
        "secondes_sur_2000_par_configuration": details,
        "secondes_sur_2000_total": round(sum(details.values()), 1),
        "extrapolation_minutes_grille_5_plis": round(total, 1),
        "note": "Extrapolation linéaire ; min_df plafonné à 2 sur 2 000 critiques pour garder un vocabulaire utile.",
    }


def mesurer_fuite_vectoriseur(textes: np.ndarray, y: np.ndarray, reference_cv: dict) -> dict:
    """Illustre la fuite sur l'entraînement seul : vectoriseur ajusté avant les plis ou dans chaque pli."""
    reglages = dict(REFERENCE)
    vectoriseur_global = f.creer_tfidf(**reglages).fit(textes)
    vocabulaire_global = set(vectoriseur_global.vocabulary_)
    matrice = vectoriseur_global.transform(textes)
    plis = list(StratifiedKFold(n_splits=N_PLIS, shuffle=True, random_state=GRAINE).split(textes, y))

    def un_pli(indices_fit, indices_val):
        modele = f.creer_classifieur("regression_logistique").fit(matrice[indices_fit], y[indices_fit])
        vocabulaire_pli = f.creer_bow(ngram_range=reglages["ngram_range"], min_df=reglages["min_df"]).fit(
            textes[indices_fit]
        ).vocabulary_
        return (
            float(accuracy_score(y[indices_val], modele.predict(matrice[indices_val]))),
            len(vocabulaire_global - set(vocabulaire_pli)),
        )

    resultats = Parallel(n_jobs=N_PROCESSUS, backend="threading")(delayed(un_pli)(a, v) for a, v in plis)
    avec_fuite = {
        "exactitude_par_pli": [round(r[0], 5) for r in resultats],
        "exactitude": round(float(np.mean([r[0] for r in resultats])), 5),
    }
    ecart = decider(reference_cv, avec_fuite)
    return {
        "protocole": "entraînement seul : TF-IDF ajusté avant les plis (mauvais) contre dans chaque pli (Pipeline)",
        "exactitude_dans_pipeline": reference_cv["exactitude"],
        "exactitude_vectoriseur_ajuste_avant_les_plis": avec_fuite["exactitude"],
        "ecart_points": ecart["gain_exactitude_points"],
        "erreur_type_points": ecart["erreur_type_points"],
        "termes_vus_seulement_grace_au_pli_de_validation": int(np.mean([r[1] for r in resultats])),
        "termes_du_vocabulaire_global": len(vocabulaire_global),
    }


# --- Topics (usage descriptif) ------------------------------------------------------------------

def mesurer_topics(recalculer: bool = False) -> dict:
    """Vocabulaire du pipeline B sur tout le corpus dédoublonné, sans aucune étiquette."""
    donnees = f.charger_textes_topics()
    textes = donnees["texte_b"].to_numpy(dtype=object)
    n_docs = len(textes)
    vectoriseur = f.creer_bow()
    matrice = vectoriseur.fit_transform(textes).tocsc()
    df = np.diff(matrice.indptr)
    occurrences = np.asarray(matrice.sum(axis=0)).ravel()

    lignes = []
    for v in GRILLE_MIN_DF:
        ligne = _ligne_vocabulaire(df, occurrences, n_docs, v, MAX_DF_TOPICS)
        garde = _masque_termes(df, n_docs, v, MAX_DF_TOPICS)
        ligne["documents_vides"] = int((matrice[:, garde].tocsr().getnnz(axis=1) == 0).sum())
        lignes.append(ligne)
    admissibles = [
        l for l in lignes if l["occurrences_conservees_pct"] >= SEUIL_OCCURRENCES_TOPICS and l["documents_vides"] == 0
    ]
    if not admissibles:
        raise ValueError("Aucun min_df ne respecte la règle fixée pour les topics.")
    min_df_retenu = max(l["min_df"] for l in admissibles)
    if f.REGLAGES_TOPICS["min_df"] != min_df_retenu or f.REGLAGES_TOPICS["max_df"] != MAX_DF_TOPICS:
        raise ValueError(f"REGLAGES_TOPICS doit valoir min_df={min_df_retenu}, max_df={MAX_DF_TOPICS}.")

    effet_max_df = [
        {"max_df": v, "termes_retires": int((df[df >= min_df_retenu] > v * n_docs).sum())}
        for v in (0.5, 0.3, 0.2, 0.1)
    ]
    frequences = df / n_docs
    plus_frequents = np.argsort(-df)[:6]
    noms = vectoriseur.get_feature_names_out()

    # Matrices prêtes pour la phase 4 : même vocabulaire pour LDA et NMF.
    bow = f.creer_vectoriseur_topics("lda")
    tfidf = f.creer_vectoriseur_topics("nmf")
    x_bow = bow.fit_transform(textes)
    x_tfidf = tfidf.fit_transform(textes)
    if list(bow.get_feature_names_out()) != list(tfidf.get_feature_names_out()):
        raise ValueError("LDA et NMF doivent partager le même vocabulaire.")
    DOSSIER_CACHE.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(DOSSIER_CACHE / "topics_bow.npz", x_bow)
    sparse.save_npz(DOSSIER_CACHE / "topics_tfidf.npz", x_tfidf)
    pd.DataFrame({"id_brut": donnees["id_brut"]}).to_parquet(DOSSIER_CACHE / "topics_ids.parquet", index=False)
    _ecrire_json("topics_vocabulaire.json", {"termes": bow.get_feature_names_out().tolist()})

    return {
        "usage": "descriptif : tout le corpus dédoublonné, sans étiquette. Interdit comme entrée d'un classifieur évalué sur le test.",
        "documents": n_docs,
        "regle_min_df": f"plus grand min_df gardant au moins {SEUIL_OCCURRENCES_TOPICS} % des occurrences et aucun document vide",
        "effet_min_df": lignes,
        "effet_max_df": effet_max_df,
        "frequence_documentaire_maximale": [
            {"terme": str(noms[i]), "part_des_documents_pct": round(100 * float(frequences[i]), 2)} for i in plus_frequents
        ],
        "reglages_retenus": {k: list(v) if isinstance(v, tuple) else v for k, v in f.REGLAGES_TOPICS.items()},
        "matrice_lda_bow": f.profiler_matrice(x_bow),
        "matrice_nmf_tfidf": f.profiler_matrice(x_tfidf),
        "fichiers": [
            "data/interim/phase3/topics_bow.npz", "data/interim/phase3/topics_tfidf.npz",
            "data/interim/phase3/topics_ids.parquet", "data/interim/phase3/topics_vocabulaire.json",
        ],
    }


# --- Figures : texte de 16 pt minimum -----------------------------------------------------------

def figure_vocabulaire(vocabulaire: dict) -> plt.Figure:
    """Effet de min_df (vocabulaire) et de max_df (occurrences gardées et mot « not »)."""
    ex.appliquer_style()
    fig, (gauche, droite) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.15, 1]})
    positions = np.arange(len(GRILLE_MIN_DF))
    for cle, couleur in RAMPE_NGRAM.items():
        termes = [l["termes"] for l in vocabulaire[cle]["effet_min_df"]]
        gauche.plot(positions, termes, color=couleur, linewidth=2.5, marker="o", markersize=9,
                    markeredgecolor=ex.SURFACE, markeredgewidth=2, label=LIBELLES_NGRAM[cle])
        gauche.annotate(ex.fr(termes[0]), (positions[0], termes[0]), xytext=(-12, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=16, color=ex.ENCRE)
        gauche.annotate(ex.fr(termes[-1]), (positions[-1], termes[-1]),
                        xytext=(12, {"1_1": 0, "1_2": -10, "1_3": 10}[cle]),
                        textcoords="offset points", ha="left", va="center", fontsize=16, color=ex.ENCRE)
    gauche.set_yscale("log")
    gauche.set_ylim(top=gauche.get_ylim()[1] * 2.6)
    gauche.set_xlim(-1.45, len(GRILLE_MIN_DF) - 1 + 1.05)
    gauche.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ex.fr(v)))
    gauche.yaxis.set_minor_formatter(NullFormatter())
    gauche.set_xticks(positions, [str(v) for v in GRILLE_MIN_DF])
    gauche.set_xlabel("min_df : nombre minimal de critiques contenant le terme")
    gauche.set_ylabel("Taille du vocabulaire (échelle log)")
    gauche.set_title("min_df réduit fortement le vocabulaire", loc="left", fontweight="bold")
    gauche.grid(axis="y")
    gauche.legend(frameon=False, loc="upper right")

    lignes = vocabulaire["1_2"]["effet_max_df"]
    etiquettes = [f"max_df = {ex.fr(l['max_df'], 1)}" for l in lignes]
    valeurs = [l["occurrences_conservees_pct"] for l in lignes]
    y = np.arange(len(lignes))
    droite.barh(y, valeurs, color=COULEUR_TFIDF, height=0.62)
    for position, ligne in zip(y, lignes):
        note = "" if ligne["not_conserve"] else " · « not » perdu"
        droite.text(ligne["occurrences_conservees_pct"] + 1, position,
                    f"{ex.fr(ligne['occurrences_conservees_pct'], 1)} %{note}", va="center", fontsize=16, color=ex.ENCRE)
    droite.set_yticks(y, etiquettes)
    droite.invert_yaxis()
    droite.set_xlim(0, 165)
    droite.set_xticks([0, 25, 50, 75, 100])
    droite.set_xlabel("Occurrences conservées (%)")
    droite.set_title("max_df bas retire « not » isolé", loc="left", fontweight="bold")
    fig.text(0.01, -0.02, "Source : IMDB Dataset (Kaggle), entraînement, pipeline A. Droite : (1,2), min_df = 5 ; "
             "les bigrammes comme « not good » restent.", fontsize=16, color=ex.ENCRE_2)
    fig.tight_layout()
    return ex._sauver(fig, "06_vocabulaire_min_df_max_df.png")


def figure_effet_reglages(decisions: dict, reference: dict) -> plt.Figure:
    """Écart d'exactitude apparié à la référence, avec un intervalle de deux erreurs-types."""
    ex.appliquer_style()
    noms = [n for n in LIBELLES_VARIANTES if n in decisions]
    fig, ax = plt.subplots(figsize=(14, 9))
    y = np.arange(len(noms))
    for position, nom in zip(y, noms):
        d = decisions[nom]
        significatif = d["ameliore"] or d["degrade"]
        ax.errorbar(d["gain_exactitude_points"], position, xerr=2 * d["erreur_type_points"],
                    color=ex.ENCRE_2, linewidth=2, capsize=4, zorder=2)
        ax.scatter(d["gain_exactitude_points"], position, s=125, zorder=3, linewidths=2,
                   facecolor=COULEUR_TFIDF if significatif else ex.SURFACE, edgecolor=COULEUR_TFIDF)
    ax.axvline(0, color=ex.ENCRE, linewidth=1.2)
    ax.set_yticks(y, [LIBELLES_VARIANTES[n] for n in noms])
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_xlabel(f"Écart d'exactitude en points (référence : {ex.fr(100 * reference['exactitude'], 2)} %)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.1f}".replace(".", ",") if v else "0"))
    ax.set_title("Effet de chaque réglage sur l'exactitude", loc="left", fontweight="bold", pad=46)
    ax.scatter([], [], s=125, facecolor=COULEUR_TFIDF, edgecolor=COULEUR_TFIDF, label="Dépasse le bruit")
    ax.scatter([], [], s=125, facecolor=ex.SURFACE, edgecolor=COULEUR_TFIDF, linewidths=2, label="Dans le bruit")
    ax.legend(frameon=False, loc="lower left", ncols=2, bbox_to_anchor=(0, 1.0), borderaxespad=0)
    fig.text(0.01, -0.03, "Validation croisée à 5 plis sur l'entraînement. Barres : ± 2 erreurs-types. Source : IMDB Dataset (Kaggle).",
             fontsize=16, color=ex.ENCRE_2)
    fig.tight_layout()
    return ex._sauver(fig, "07_effet_reglages_vectoriseur.png")


def figure_bow_tfidf(comparaison: dict) -> plt.Figure:
    """Exactitude de BoW et TF-IDF pour chaque modèle supervisé."""
    ex.appliquer_style()
    modeles = list(comparaison)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    y = np.arange(len(modeles))
    for position, modele in zip(y, modeles):
        bow = 100 * comparaison[modele]["bow"]["exactitude"]
        tfidf = 100 * comparaison[modele]["tfidf"]["exactitude"]
        ax.plot([bow, tfidf], [position, position], color=ex.AXE, linewidth=3, zorder=1)
        ax.scatter(bow, position, s=190, color=COULEUR_BOW, edgecolor=ex.SURFACE, linewidth=2, zorder=3)
        ax.scatter(tfidf, position, s=190, color=COULEUR_TFIDF, edgecolor=ex.SURFACE, linewidth=2, zorder=3)
        gauche_bow = bow < tfidf
        ax.annotate(f"BoW {ex.fr(bow, 2)} %", (bow, position), xytext=(-13 if gauche_bow else 13, 0),
                    textcoords="offset points", ha="right" if gauche_bow else "left", va="center", fontsize=16)
        ax.annotate(f"TF-IDF {ex.fr(tfidf, 2)} %", (tfidf, position), xytext=(13 if gauche_bow else -13, 0),
                    textcoords="offset points", ha="left" if gauche_bow else "right", va="center", fontsize=16)
    ax.set_yticks(y, [LIBELLES_MODELES[m] for m in modeles])
    ax.invert_yaxis()
    valeurs = [100 * comparaison[m][n]["exactitude"] for m in modeles for n in ("bow", "tfidf")]
    ax.set_xlim(min(valeurs) - 1.6, max(valeurs) + 1.6)
    ax.grid(axis="x")
    ax.set_xlabel("Exactitude moyenne sur 5 plis (%), axe tronqué")
    ax.set_title("BoW contre TF-IDF, sur le même vocabulaire", loc="left", fontweight="bold", pad=46)
    ax.scatter([], [], s=190, color=COULEUR_BOW, label="BoW (comptages)")
    ax.scatter([], [], s=190, color=COULEUR_TFIDF, label="TF-IDF (sublinear_tf)")
    ax.legend(frameon=False, loc="lower left", ncols=2, bbox_to_anchor=(0, 1.0), borderaxespad=0)
    fig.text(0.01, -0.04, "Validation croisée sur l'entraînement, (1,2), min_df = 5. Source : IMDB Dataset (Kaggle).",
             fontsize=16, color=ex.ENCRE_2)
    fig.tight_layout()
    return ex._sauver(fig, "08_bow_contre_tfidf.png")


# --- Orchestration ------------------------------------------------------------------------------

def executer(recalculer: bool = False) -> dict:
    """Enchaîne la phase 3, à partir des caches, et met à jour la clé `vectorisation`."""
    DOSSIER_CACHE.mkdir(parents=True, exist_ok=True)
    separation = controler_separation(recalculer)
    print("Séparation :", {k: separation[k] for k in ("entrainement", "test", "index_communs", "textes_communs")})

    train = f.charger_entrainement()
    textes = train["texte_a"].to_numpy(dtype=object)
    y = (train["sentiment"] == "positive").astype(int).to_numpy()

    chrono = None if recalculer else _lire_json("chrono.json")
    if chrono is None:
        chrono = chronometrer(textes, y)
        _ecrire_json("chrono.json", chrono)
    print("Chronométrage :", chrono["extrapolation_minutes_grille_5_plis"], "min extrapolées")

    cv = {} if recalculer else (_lire_json("validation_croisee.json") or {})
    for nom in [*VARIANTES_TFIDF, *MODELES]:
        if nom not in cv:
            cv[nom] = evaluer_config(nom, textes, y)
            _ecrire_json("validation_croisee.json", cv)
            print(f"  CV {nom} : {cv[nom]['exactitude']:.5f} en {cv[nom]['duree_s']} s", flush=True)

    reference = cv["reference"]
    decisions = {n: decider(reference, cv[n]) for n in VARIANTES_TFIDF if n not in VARIANTES_HORS_FIGURE}
    # Même retrait de « not » par max_df = 0,7, mesuré sans bigrammes pour isoler son effet.
    effet_not_sans_bigrammes = decider(cv["unigrammes"], cv["unigrammes_max_df_0.7"])
    comparaison = {}
    for modele in ("naive_bayes", "regression_logistique", "svm_lineaire"):
        tfidf = reference if modele == "regression_logistique" else cv[f"{modele}_tfidf"]
        bow = cv[f"{modele}_bow"]
        comparaison[modele] = {
            "bow": {k: bow[k] for k in ("exactitude", "f1", "roc_auc", "exactitude_par_pli", "alertes_convergence", "duree_s")},
            "tfidf": {k: tfidf[k] for k in ("exactitude", "f1", "roc_auc", "exactitude_par_pli", "alertes_convergence", "duree_s")},
            "tfidf_moins_bow": decider(bow, tfidf),
        }

    fuite = None if recalculer else _lire_json("fuite_vectoriseur.json")
    if fuite is None:
        fuite = mesurer_fuite_vectoriseur(textes, y, reference)
        _ecrire_json("fuite_vectoriseur.json", fuite)

    vocabulaire = None if recalculer else _lire_json("vocabulaire_a.json")
    if vocabulaire is None:
        vocabulaire = {
            "par_ngram": mesurer_vocabulaire_a(textes),
            "matrice_reference": mesurer_matrice_reference(textes),
        }
        _ecrire_json("vocabulaire_a.json", vocabulaire)

    topics = None if recalculer else _lire_json("topics.json")
    if topics is None:
        topics = mesurer_topics(recalculer)
        _ecrire_json("topics.json", topics)

    figure_vocabulaire(vocabulaire["par_ngram"])
    figure_effet_reglages(decisions, reference)
    figure_bow_tfidf(comparaison)

    bilan = {
        "separation": separation,
        "chronometrage": chrono,
        "vocabulaire_pipeline_a": vocabulaire,
        "validation_croisee": {
            "plis": N_PLIS,
            "perimetre": "entraînement seul ; vectoriseur ajusté dans chaque pli",
            "regle": "écart retenu si gain moyen apparié >= 0,05 point et > 2 erreurs-types",
            "resultats": cv,
            "decisions_reglages": decisions,
            "effet_max_df_0_7_sans_bigrammes": effet_not_sans_bigrammes,
            "pistes_pour_la_phase_5": [n for n, d in decisions.items() if d["ameliore"]],
            "avertissement": (
                "Douze réglages sont comparés sur les mêmes plis : un gain isolé peut venir du hasard. "
                "Ces pistes ne sont pas retenues ici ; la phase 5 les règle dans sa recherche."
            ),
            "bow_contre_tfidf": comparaison,
            "reglages_de_depart_phase_5": {
                k: list(v) if isinstance(v, tuple) else v for k, v in REFERENCE.items()
            },
        },
        "fuite_vectoriseur_illustration": fuite,
        "topics_pipeline_b": topics,
        "association_modeles": {"LDA": "BoW", "Naive Bayes": "BoW", "NMF": "TF-IDF",
                                "Régression logistique": "TF-IDF", "SVM linéaire": "TF-IDF"},
    }
    ex.enregistrer_metrics("vectorisation", bilan)
    return bilan


if __name__ == "__main__":
    analyseur = argparse.ArgumentParser(description="Mesures et caches de la phase 3.")
    analyseur.add_argument("--recalculer", action="store_true", help="Ignore les caches existants.")
    executer(recalculer=analyseur.parse_args().recalculer)

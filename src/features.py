"""Vectorisation des critiques : sac de mots (BoW), TF-IDF et pipelines supervisés.

- BoW sert à LDA et à Naive Bayes ; TF-IDF sert à NMF, à la régression logistique et au SVM.
- Tout vectoriseur supervisé s'ajuste dans un `Pipeline`, sur l'entraînement seul.
- Les vectoriseurs de topics sont descriptifs : ils peuvent lire tout le corpus dédoublonné,
  mais ne doivent jamais alimenter un classifieur évalué sur le test.

Ce module est importé par le notebook et par l'application. Il ne contient aucun calcul long.

Exemple d'utilisation :
    from src.features import charger_entrainement, creer_pipeline_supervise
    train = charger_entrainement()
    modele = creer_pipeline_supervise("regression_logistique")
    modele.fit(train["texte_a"], (train["sentiment"] == "positive").astype(int))
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing import MOTIF_VECTORISEUR_A, RACINE

GRAINE = 42
CHEMIN_TEXTES = RACINE / "data" / "interim" / "textes_pretraites.parquet"
CHEMIN_INDEX = RACINE / "data" / "interim" / "separation_80_20.parquet"
PARTITION_TRAIN, PARTITION_TEST = "entrainement", "test"

# Réglage de comparaison, identique à la référence de la phase 2.
REGLAGES_SUPERVISES = {"ngram_range": (1, 2), "min_df": 5, "max_df": 1.0}

# Topics : même vocabulaire pour LDA et NMF. min_df suit la règle de src/features_eval.py.
REGLAGES_TOPICS = {"ngram_range": (1, 1), "min_df": 10, "max_df": 0.5}

# Association imposée par le cahier : quel vectoriseur alimente quel modèle.
VECTORISEUR_PAR_MODELE = {
    "naive_bayes": "bow",
    "regression_logistique": "tfidf",
    "svm_lineaire": "tfidf",
}


# --- Fabriques de vectoriseurs ---------------------------------------------------------------

def creer_vectoriseur(
    nature: str = "tfidf",
    *,
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int | float = 1,
    max_df: int | float = 1.0,
    sublinear_tf: bool = False,
    max_features: int | None = None,
) -> CountVectorizer | TfidfVectorizer:
    """Crée un vectoriseur pour des textes déjà prétraités, découpés par des espaces.

    `nature` vaut "bow" ou "tfidf". `sublinear_tf` n'a de sens que pour TF-IDF.
    """
    communs = {
        "token_pattern": MOTIF_VECTORISEUR_A,  # les jetons sont déjà formés par src/preprocessing
        "lowercase": False,
        "ngram_range": tuple(ngram_range),
        "min_df": min_df,
        "max_df": max_df,
        "max_features": max_features,
    }
    if nature == "bow":
        if sublinear_tf:
            raise ValueError("sublinear_tf ne s'applique qu'à TF-IDF.")
        return CountVectorizer(**communs)
    if nature == "tfidf":
        return TfidfVectorizer(sublinear_tf=sublinear_tf, **communs)
    raise ValueError("nature doit valoir 'bow' ou 'tfidf'.")


def creer_bow(**reglages) -> CountVectorizer:
    """Sac de mots : compte les occurrences de chaque terme."""
    return creer_vectoriseur("bow", **reglages)


def creer_tfidf(**reglages) -> TfidfVectorizer:
    """TF-IDF : pondère chaque terme selon sa rareté dans les documents."""
    return creer_vectoriseur("tfidf", **reglages)


def creer_vectoriseur_topics(modele: str) -> CountVectorizer | TfidfVectorizer:
    """Vectoriseur descriptif du pipeline B : BoW pour "lda", TF-IDF pour "nmf"."""
    if modele == "lda":
        return creer_bow(**REGLAGES_TOPICS)
    if modele == "nmf":
        return creer_tfidf(**REGLAGES_TOPICS)
    raise ValueError("modele doit valoir 'lda' ou 'nmf'.")


# --- Pipelines supervisés --------------------------------------------------------------------

def creer_classifieur(nom: str):
    """Classifieur aux réglages par défaut. Les hyperparamètres sont réglés en phase 5."""
    if nom == "naive_bayes":
        return MultinomialNB(alpha=1.0)
    if nom == "regression_logistique":
        return LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=GRAINE)
    if nom == "svm_lineaire":
        # 5 000 itérations ne suffisaient pas sur un pli avec BoW (alerte observée en phase 3).
        return LinearSVC(C=1.0, dual="auto", max_iter=20000, random_state=GRAINE)
    raise ValueError(f"Modèle inconnu : {nom}")


def creer_pipeline_supervise(
    modele: str, nature: str | None = None, **reglages_vectoriseur
) -> Pipeline:
    """Assemble vectoriseur et classifieur : `fit` n'apprend le vocabulaire que sur ses données.

    Sans `nature`, le vectoriseur est celui prévu par le cahier pour ce modèle.
    """
    if modele not in VECTORISEUR_PAR_MODELE:
        raise ValueError(f"Modèle inconnu : {modele}")
    nature = nature or VECTORISEUR_PAR_MODELE[modele]
    reglages = {**REGLAGES_SUPERVISES, **reglages_vectoriseur}
    if nature == "tfidf":
        reglages.setdefault("sublinear_tf", True)
    return Pipeline([
        ("vectoriseur", creer_vectoriseur(nature, **reglages)),
        ("classifieur", creer_classifieur(modele)),
    ])


# --- Séparation et lecture des textes --------------------------------------------------------

def empreinte_index(identifiants: np.ndarray | pd.Series) -> str:
    """Empreinte SHA256 d'une liste d'identifiants, dans l'ordre croissant."""
    valeurs = np.sort(np.asarray(identifiants, dtype=np.int64))
    return hashlib.sha256(valeurs.tobytes()).hexdigest().upper()


def sauvegarder_index(
    id_train: np.ndarray | pd.Series,
    id_test: np.ndarray | pd.Series,
    chemin: str | Path = CHEMIN_INDEX,
) -> Path:
    """Sauvegarde les identifiants de la séparation, après avoir vérifié qu'ils sont disjoints."""
    id_train, id_test = np.asarray(id_train), np.asarray(id_test)
    if np.intersect1d(id_train, id_test).size:
        raise ValueError("Une même critique figure dans l'entraînement et dans le test.")
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "id_brut": np.concatenate([id_train, id_test]).astype(np.int64),
        "partition": [PARTITION_TRAIN] * len(id_train) + [PARTITION_TEST] * len(id_test),
    }).sort_values("id_brut").to_parquet(chemin, index=False)
    return chemin


def charger_index(chemin: str | Path = CHEMIN_INDEX) -> tuple[np.ndarray, np.ndarray]:
    """Relit les identifiants d'entraînement et de test sauvegardés."""
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Index de séparation introuvables : {chemin}")
    index = pd.read_parquet(chemin)
    return (
        index.loc[index["partition"] == PARTITION_TRAIN, "id_brut"].to_numpy(),
        index.loc[index["partition"] == PARTITION_TEST, "id_brut"].to_numpy(),
    )


def charger_entrainement(
    colonnes: tuple[str, ...] = ("id_brut", "sentiment", "texte_a"),
    chemin: str | Path = CHEMIN_TEXTES,
) -> pd.DataFrame:
    """Lit uniquement les lignes d'entraînement : les étiquettes du test ne sont pas chargées."""
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Textes prétraités introuvables : {chemin}")
    return pd.read_parquet(
        chemin, columns=list(colonnes), filters=[("partition", "==", PARTITION_TRAIN)]
    ).reset_index(drop=True)


def ajuster_sur_entrainement(vectoriseur, donnees: pd.DataFrame, colonne: str = "texte_a"):
    """Ajuste un vectoriseur sur les seules lignes d'entraînement d'un tableau complet."""
    if "partition" not in donnees.columns:
        raise ValueError("La colonne `partition` est nécessaire pour isoler l'entraînement.")
    masque = donnees["partition"] == PARTITION_TRAIN
    if not masque.any():
        raise ValueError("Aucune ligne d'entraînement dans ce tableau.")
    return vectoriseur.fit(donnees.loc[masque, colonne])


def charger_textes_topics(chemin: str | Path = CHEMIN_TEXTES) -> pd.DataFrame:
    """Lit les textes du pipeline B de tout le corpus, sans aucune étiquette (usage descriptif)."""
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Textes prétraités introuvables : {chemin}")
    return pd.read_parquet(chemin, columns=["id_brut", "texte_b"])


# --- Description d'une matrice ---------------------------------------------------------------

def profiler_matrice(matrice: sparse.spmatrix) -> dict:
    """Résume une matrice documents-termes : dimensions, creux et mémoire."""
    matrice = sparse.csr_matrix(matrice)
    documents, termes = matrice.shape
    cases = documents * termes
    non_nuls = int(matrice.nnz)
    octets = matrice.data.nbytes + matrice.indices.nbytes + matrice.indptr.nbytes
    return {
        "documents": int(documents),
        "termes": int(termes),
        "valeurs_non_nulles": non_nuls,
        "taux_de_creux_pct": round(100 * (1 - non_nuls / cases), 4) if cases else 100.0,
        "termes_non_nuls_par_document": round(non_nuls / documents, 1) if documents else 0.0,
        "memoire_creuse_mo": round(octets / 1024 ** 2, 1),
        "memoire_dense_theorique_go": round(cases * 8 / 1024 ** 3, 1),
    }

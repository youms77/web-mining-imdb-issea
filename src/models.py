"""Modèles de sentiment : référence VADER, Naive Bayes, régression logistique, SVM linéaire.

- Tout modèle supervisé est un `Pipeline` : vocabulaire, IDF et classifieur n'apprennent que sur
  les données passées à `fit`.
- La calibration enveloppe le `Pipeline` complet : le vectoriseur est réappris dans chaque pli interne.
- `FiltreTermes` reproduit `min_df` et `max_df` sur une matrice de comptages déjà calculée. Il sert
  à accélérer la recherche d'hyperparamètres, sans changer le résultat (voir tests/test_models.py).

Exemple d'utilisation :
    from src.models import creer_pipeline_texte, calibrer
    modele = calibrer(creer_pipeline_texte("regression_logistique", {"C": 10}), "sigmoid")
    modele.fit(textes_a, y)
    probabilites = modele.predict_proba(["great movie"])[:, 1]
"""
from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src import features as f
from src.preprocessing import nettoyer_html, pipeline_a, retirer_notes

GRAINE = 42
N_PLIS_CALIBRATION = 5

# Famille -> (classifieur, vectoriseur). Naive Bayes est testé avec les deux vectoriseurs.
FAMILLES = {
    "naive_bayes_bow": ("naive_bayes", "bow"),
    "naive_bayes_tfidf": ("naive_bayes", "tfidf"),
    "regression_logistique": ("regression_logistique", "tfidf"),
    "svm_lineaire": ("svm_lineaire", "tfidf"),
}
LIBELLES_FAMILLES = {
    "vader": "VADER (lexique)",
    "naive_bayes_bow": "Naive Bayes, BoW",
    "naive_bayes_tfidf": "Naive Bayes, TF-IDF",
    "regression_logistique": "Régression logistique",
    "svm_lineaire": "SVM linéaire",
}
# Paramètre réglé pour chaque classifieur, du plus régularisé au moins régularisé.
GRILLE_CLASSIFIEUR = {
    "naive_bayes": ("alpha", (1.0, 0.3, 0.1, 0.03)),
    # Grilles étendues : la première version s'arrêtait à C=30 et C=1, valeurs choisies au bord.
    "regression_logistique": ("C", (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)),
    "svm_lineaire": ("C", (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)),
}
# min_df = 2 et 3 ajoutés : à C élevé, 5 battait 10, donc 5 était lui aussi un bord de grille.
GRILLE_MIN_DF = (2, 3, 5, 10, 20)
GRILLE_MAX_DF = (1.0, 0.5, 0.3)
NGRAM = (1, 2)


class FiltreTermes(TransformerMixin, BaseEstimator):
    """Applique `min_df` et `max_df` à une matrice de comptages, avec la règle de scikit-learn."""

    def __init__(self, min_df: int | float = 5, max_df: int | float = 1.0):
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, X, y=None):
        X = sparse.csc_matrix(X)
        n_docs = X.shape[0]
        haut = self.max_df if isinstance(self.max_df, Integral) else self.max_df * n_docs
        bas = self.min_df if isinstance(self.min_df, Integral) else self.min_df * n_docs
        df = X.getnnz(axis=0)
        self.indices_ = np.flatnonzero((df >= bas) & (df <= haut))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return sparse.csr_matrix(X)[:, self.indices_]


def _regler_classifieur(pipeline: Pipeline, hyperparametres: dict) -> Pipeline:
    return pipeline.set_params(**{f"classifieur__{k}": v for k, v in hyperparametres.items()})


def creer_pipeline_texte(
    famille: str, hyperparametres: dict | None = None, min_df: int = 5, max_df: float = 1.0
) -> Pipeline:
    """Pipeline complet sur le texte du pipeline A : vectoriseur puis classifieur."""
    modele, nature = FAMILLES[famille]
    pipeline = f.creer_pipeline_supervise(modele, nature, ngram_range=NGRAM, min_df=min_df, max_df=max_df)
    return _regler_classifieur(pipeline, hyperparametres or {})


def creer_pipeline_comptages(
    famille: str, hyperparametres: dict | None = None, min_df: int = 5, max_df: float = 1.0
) -> Pipeline:
    """Même modèle que `creer_pipeline_texte`, à partir de comptages de mots et bigrammes."""
    modele, nature = FAMILLES[famille]
    etapes = [("filtre", FiltreTermes(min_df=min_df, max_df=max_df))]
    if nature == "tfidf":
        etapes.append(("ponderation", TfidfTransformer(sublinear_tf=True)))
    etapes.append(("classifieur", f.creer_classifieur(modele)))
    return _regler_classifieur(Pipeline(etapes), hyperparametres or {})


def preparer_texte(critique_brute: object) -> str:
    """Prétraitement exact attendu par les modèles gelés : notes écrites retirées, puis pipeline A."""
    if not isinstance(critique_brute, str):
        return ""
    return pipeline_a(retirer_notes(critique_brute))


def plis_calibration() -> StratifiedKFold:
    """Plis internes de la calibration, fixés par la graine."""
    return StratifiedKFold(n_splits=N_PLIS_CALIBRATION, shuffle=True, random_state=GRAINE)


def calibrer(pipeline: Pipeline, methode: str = "sigmoid", n_jobs: int | None = None) -> CalibratedClassifierCV:
    """Calibre un Pipeline sur des scores hors pli.

    `ensemble=False` : les scores hors pli servent au calibrateur, puis un seul Pipeline est
    réappris sur toutes les données reçues. Le modèle sauvegardé garde donc un seul vocabulaire.
    """
    if methode not in ("sigmoid", "isotonic"):
        raise ValueError("methode doit valoir 'sigmoid' ou 'isotonic'.")
    return CalibratedClassifierCV(
        estimator=pipeline, method=methode, cv=plis_calibration(), ensemble=False, n_jobs=n_jobs
    )


def score_continu(modele, X) -> np.ndarray:
    """Score de classement : fonction de décision si elle existe, sinon probabilité positive."""
    if hasattr(modele, "decision_function"):
        return np.asarray(modele.decision_function(X))
    return modele.predict_proba(X)[:, 1]


class ReferenceVader:
    """Référence lexicale : aucun apprentissage, seuil fixé à 0 sur le score `compound`.

    VADER lit la casse et la ponctuation : il reçoit la critique brute sans HTML, pas le pipeline A.
    Son score va de -1 à 1 et n'est pas une probabilité.
    """

    seuil = 0.0

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyseur = SentimentIntensityAnalyzer()

    def scores(self, textes) -> np.ndarray:
        return np.array([self._analyseur.polarity_scores(nettoyer_html(t))["compound"] for t in textes])

    def predire(self, scores: np.ndarray) -> np.ndarray:
        return (np.asarray(scores) >= self.seuil).astype(int)


def sauvegarder_modele(modele, chemin: str | Path) -> dict:
    """Sauvegarde compressée avec joblib. Renvoie la taille et l'empreinte du fichier."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modele, chemin, compress=("xz", 3))
    return {
        "fichier": chemin.name,
        "taille_mo": round(chemin.stat().st_size / 1024 ** 2, 2),
        "sha256": hashlib.sha256(chemin.read_bytes()).hexdigest().upper(),
    }


def charger_modele(chemin: str | Path):
    """Recharge un modèle sauvegardé par `sauvegarder_modele`."""
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Modèle introuvable : {chemin}")
    return joblib.load(chemin)

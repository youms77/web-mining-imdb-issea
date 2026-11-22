"""Recherche d'hyperparamètres de la phase 5, sur l'entraînement seul.

Pour chaque pli externe, les comptages (mots et bigrammes, plus petit min_df de la grille) sont
appris sur la partie d'ajustement seule, puis mis en cache. `FiltreTermes` en déduit chaque couple
(min_df, max_df) : tout terme retenu pour un min_df de la grille figure forcément dans ces comptages.

Aucune étiquette ni aucun texte du test n'est chargé ici.

Exemple d'utilisation :
    python -m src.sentiment_grille --etape chrono
    python -m src.sentiment_grille --etape grille
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src import features as f
from src import models as m
from src.preprocessing import RACINE

GRAINE = 42
N_PLIS = 5
N_PROCESSUS = 4
DOSSIER = RACINE / "data" / "interim" / "phase5"
DOSSIER_PLIS = DOSSIER / "plis"
DOSSIER_GRILLE = DOSSIER / "grille"


def lire_entrainement() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identifiants, textes du pipeline A et étiquettes de l'entraînement."""
    train = f.charger_entrainement()
    return (
        train["id_brut"].to_numpy(),
        train["texte_a"].to_numpy(dtype=object),
        (train["sentiment"] == "positive").astype(int).to_numpy(),
    )


def plis_externes(textes: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Mêmes 5 plis qu'aux phases 2 et 3."""
    return list(StratifiedKFold(n_splits=N_PLIS, shuffle=True, random_state=GRAINE).split(textes, y))


def _preparer_pli(k: int, textes, indices_fit, indices_val, dossier) -> float:
    debut = time.perf_counter()
    vectoriseur = f.creer_bow(ngram_range=m.NGRAM, min_df=min(m.GRILLE_MIN_DF))
    x_fit = vectoriseur.fit_transform(textes[indices_fit])
    x_val = vectoriseur.transform(textes[indices_val])
    sparse.save_npz(dossier / f"pli{k}_fit.npz", x_fit)
    sparse.save_npz(dossier / f"pli{k}_val.npz", x_val)
    return time.perf_counter() - debut


def preparer_plis(textes, y, dossier=DOSSIER_PLIS) -> None:
    """Calcule et met en cache les comptages de chaque pli, s'ils n'existent pas déjà."""
    dossier.mkdir(parents=True, exist_ok=True)
    plis = plis_externes(textes, y)
    a_faire = [k for k in range(N_PLIS) if not (dossier / f"pli{k}_val.npz").exists()]
    if a_faire:
        durees = Parallel(n_jobs=3)(
            delayed(_preparer_pli)(k, textes, plis[k][0], plis[k][1], dossier) for k in a_faire
        )
        print("Comptages par pli (s) :", [round(d, 1) for d in durees], flush=True)


def _cle(k: int, min_df: int, max_df: float) -> str:
    return f"pli{k}_min{min_df}_max{max_df}.json"


def _job(k: int, min_df: int, max_df: float, y_fit, y_val) -> dict:
    """Évalue tous les réglages de classifieur pour un pli et un couple (min_df, max_df)."""
    debut = time.perf_counter()
    x_fit = sparse.load_npz(DOSSIER_PLIS / f"pli{k}_fit.npz")
    x_val = sparse.load_npz(DOSSIER_PLIS / f"pli{k}_val.npz")
    filtre = m.FiltreTermes(min_df=min_df, max_df=max_df).fit(x_fit)
    bow_fit, bow_val = filtre.transform(x_fit), filtre.transform(x_val)
    ponderation = m.TfidfTransformer(sublinear_tf=True).fit(bow_fit)
    tfidf_fit, tfidf_val = ponderation.transform(bow_fit), ponderation.transform(bow_val)
    lignes = []
    for famille, (classifieur, nature) in m.FAMILLES.items():
        a, b = (bow_fit, bow_val) if nature == "bow" else (tfidf_fit, tfidf_val)
        parametre, valeurs = m.GRILLE_CLASSIFIEUR[classifieur]
        for valeur in valeurs:
            modele = f.creer_classifieur(classifieur).set_params(**{parametre: valeur})
            with warnings.catch_warnings(record=True) as alertes:
                warnings.simplefilter("always", ConvergenceWarning)
                modele.fit(a, y_fit)
            score = m.score_continu(modele, b)
            lignes.append({
                "famille": famille, "parametre": parametre, "valeur": valeur,
                "min_df": min_df, "max_df": max_df, "pli": k, "termes": int(a.shape[1]),
                "exactitude": float(accuracy_score(y_val, modele.predict(b))),
                "f1": float(f1_score(y_val, modele.predict(b))),
                "roc_auc": float(roc_auc_score(y_val, score)),
                "alertes_convergence": sum(issubclass(x.category, ConvergenceWarning) for x in alertes),
            })
    resultat = {"duree_s": round(time.perf_counter() - debut, 1), "lignes": lignes}
    temporaire = DOSSIER_GRILLE / (_cle(k, min_df, max_df) + ".tmp")
    temporaire.write_text(json.dumps(resultat), encoding="utf-8")
    temporaire.replace(DOSSIER_GRILLE / _cle(k, min_df, max_df))
    return resultat


def chronometrer(textes, y) -> dict:
    """Mesure 2 000 critiques, puis un vrai travail de grille sur un pli complet."""
    tirage = np.random.default_rng(GRAINE)
    indices = np.concatenate([
        tirage.choice(np.flatnonzero(y == c), size=1000, replace=False) for c in (0, 1)
    ])
    debut = time.perf_counter()
    x = f.creer_bow(ngram_range=m.NGRAM, min_df=2).fit_transform(textes[indices])
    duree_comptage = time.perf_counter() - debut
    debut = time.perf_counter()
    tfidf = m.TfidfTransformer(sublinear_tf=True).fit_transform(x)
    for famille, (classifieur, nature) in m.FAMILLES.items():
        parametre, valeurs = m.GRILLE_CLASSIFIEUR[classifieur]
        for valeur in valeurs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                f.creer_classifieur(classifieur).set_params(**{parametre: valeur}).fit(
                    x if nature == "bow" else tfidf, y[indices]
                )
    duree_modeles = time.perf_counter() - debut
    taille_pli = len(textes) * (N_PLIS - 1) / N_PLIS
    n_couples = len(m.GRILLE_MIN_DF) * len(m.GRILLE_MAX_DF)
    extrapolation = (
        duree_comptage * taille_pli / 2000 * N_PLIS / 3
        + duree_modeles * taille_pli / 2000 * N_PLIS * n_couples / N_PROCESSUS
    ) / 60

    preparer_plis(textes, y)
    DOSSIER_GRILLE.mkdir(parents=True, exist_ok=True)
    plis = plis_externes(textes, y)
    debut = time.perf_counter()
    # Cas le plus lourd de la grille : le plus grand vocabulaire.
    _job(0, min(m.GRILLE_MIN_DF), 1.0, y[plis[0][0]], y[plis[0][1]])
    duree_job = time.perf_counter() - debut
    n_jobs = N_PLIS * n_couples
    resultat = {
        "sur_2000_comptage_s": round(duree_comptage, 2),
        "sur_2000_tous_classifieurs_s": round(duree_modeles, 2),
        "extrapolation_lineaire_grille_min": round(extrapolation, 1),
        "travail_reel_un_pli_un_couple_s": round(duree_job, 1),
        "travaux": n_jobs,
        "processus": N_PROCESSUS,
        "prevision_grille_min": round(duree_job * n_jobs / N_PROCESSUS / 60, 1),
        "prevision_avec_marge_2_4_min": round(2.4 * duree_job * n_jobs / N_PROCESSUS / 60, 1),
    }
    (DOSSIER / "chrono.json").write_text(json.dumps(resultat, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultat


def lancer_grille(textes, y) -> None:
    """Exécute les travaux manquants ; une interruption ne perd que les travaux en cours."""
    preparer_plis(textes, y)
    DOSSIER_GRILLE.mkdir(parents=True, exist_ok=True)
    plis = plis_externes(textes, y)
    travaux = [
        (k, a, b)
        for k in range(N_PLIS) for a in m.GRILLE_MIN_DF for b in m.GRILLE_MAX_DF
        if not (DOSSIER_GRILLE / _cle(k, a, b)).exists()
    ]
    print(f"Travaux à exécuter : {len(travaux)}", flush=True)
    debut = time.perf_counter()
    Parallel(n_jobs=N_PROCESSUS, verbose=5)(
        delayed(_job)(k, a, b, y[plis[k][0]], y[plis[k][1]]) for k, a, b in travaux
    )
    print(f"Grille terminée en {(time.perf_counter() - debut) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    analyseur = argparse.ArgumentParser(description="Recherche d'hyperparamètres (entraînement seul).")
    analyseur.add_argument("--etape", choices=["chrono", "grille"], required=True)
    etape = analyseur.parse_args().etape
    _, textes_train, y_train = lire_entrainement()
    if etape == "chrono":
        print(json.dumps(chronometrer(textes_train, y_train), ensure_ascii=False, indent=1))
    else:
        lancer_grille(textes_train, y_train)

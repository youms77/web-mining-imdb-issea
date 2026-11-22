"""Mesures d'évaluation : classement, calibration, bootstrap, zone d'incertitude.

Ce module ne lit aucun fichier : il calcule à partir d'étiquettes et de probabilités.

Exemple d'utilisation :
    from src.evaluation import metriques_classement, choisir_zone_incertitude
    m = metriques_classement(y, proba)
    zone = choisir_zone_incertitude(y_validation, proba_validation, cible=0.95)
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

GRAINE = 42
N_BOOTSTRAP = 2000
N_CLASSES_FIABILITE = 10


def metriques_classement(y, score, predictions=None, seuil: float = 0.5) -> dict:
    """Exactitude, précision, rappel, F1, ROC-AUC et matrice de confusion (positif = 1)."""
    y = np.asarray(y)
    score = np.asarray(score)
    predictions = (score >= seuil).astype(int) if predictions is None else np.asarray(predictions)
    vn, fp, fn, vp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "exactitude": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "rappel": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)),
        "matrice_confusion": {"vrais_negatifs": int(vn), "faux_positifs": int(fp),
                              "faux_negatifs": int(fn), "vrais_positifs": int(vp)},
    }


def erreur_calibration(y, proba, n_classes: int = N_CLASSES_FIABILITE) -> float:
    """ECE : écart moyen, pondéré par l'effectif, entre probabilité annoncée et fréquence observée."""
    y, proba = np.asarray(y), np.asarray(proba)
    classes = np.minimum((proba * n_classes).astype(int), n_classes - 1)
    total = 0.0
    for c in range(n_classes):
        masque = classes == c
        if masque.any():
            total += masque.mean() * abs(y[masque].mean() - proba[masque].mean())
    return float(total)


def metriques_probabilite(y, proba) -> dict:
    """Qualité des probabilités : Brier, perte logarithmique et ECE (plus bas = meilleur)."""
    proba_bornee = np.clip(proba, 1e-15, 1 - 1e-15)
    return {
        "brier": float(brier_score_loss(y, proba)),
        "log_loss": float(log_loss(y, proba_bornee, labels=[0, 1])),
        "ece": erreur_calibration(y, proba),
    }


def courbe_fiabilite(y, proba, n_classes: int = N_CLASSES_FIABILITE) -> list[dict]:
    """Par tranche de probabilité : effectif, probabilité moyenne et fréquence observée."""
    y, proba = np.asarray(y), np.asarray(proba)
    classes = np.minimum((proba * n_classes).astype(int), n_classes - 1)
    lignes = []
    for c in range(n_classes):
        masque = classes == c
        if masque.any():
            lignes.append({
                "tranche": [c / n_classes, (c + 1) / n_classes],
                "effectif": int(masque.sum()),
                "probabilite_moyenne": float(proba[masque].mean()),
                "frequence_observee": float(y[masque].mean()),
            })
    return lignes


def ecart_apparie(a, b) -> dict:
    """Écart moyen a - b sur les mêmes plis, avec son erreur-type."""
    ecarts = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {
        "ecart": float(ecarts.mean()),
        "erreur_type": float(ecarts.std(ddof=1) / math.sqrt(len(ecarts))),
    }


def bootstrap_ic(y, score, predictions=None, seuil: float = 0.5, n: int = N_BOOTSTRAP,
                 graine: int = GRAINE, avec_brier: bool = True) -> dict:
    """Intervalles de confiance à 95 % par bootstrap percentile, en retirant les critiques."""
    y = np.asarray(y)
    score = np.asarray(score)
    predictions = (score >= seuil).astype(int) if predictions is None else np.asarray(predictions)
    tirage = np.random.default_rng(graine)
    taille = len(y)
    valeurs = {k: np.empty(n) for k in ("exactitude", "precision", "rappel", "f1", "roc_auc")}
    if avec_brier:
        valeurs["brier"] = np.empty(n)
    for i in range(n):
        indices = tirage.integers(0, taille, taille)
        y_b, p_b = y[indices], predictions[indices]
        vp = int(((y_b == 1) & (p_b == 1)).sum())
        fp = int(((y_b == 0) & (p_b == 1)).sum())
        fn = int(((y_b == 1) & (p_b == 0)).sum())
        valeurs["exactitude"][i] = (y_b == p_b).mean()
        valeurs["precision"][i] = vp / (vp + fp) if vp + fp else 0.0
        valeurs["rappel"][i] = vp / (vp + fn) if vp + fn else 0.0
        valeurs["f1"][i] = 2 * vp / (2 * vp + fp + fn) if vp else 0.0
        valeurs["roc_auc"][i] = roc_auc_score(y_b, score[indices])
        if avec_brier:
            valeurs["brier"][i] = np.mean((score[indices] - y_b) ** 2)
    return {
        k: {"bas": float(np.percentile(v, 2.5)), "haut": float(np.percentile(v, 97.5))}
        for k, v in valeurs.items()
    }


def choisir_zone_incertitude(y, proba, cible: float = 0.95, pas: float = 0.005) -> dict:
    """Plus petite zone symétrique autour de 0,5 telle que l'exactitude hors zone atteigne `cible`.

    Une critique est « incertaine » si bas < p < haut. À appeler sur des probabilités hors pli.
    """
    y, proba = np.asarray(y), np.asarray(proba)
    predictions = (proba >= 0.5).astype(int)
    for k in range(int(round(0.5 / pas))):
        demi_largeur = round(k * pas, 6)
        bas, haut = round(0.5 - demi_largeur, 6), round(0.5 + demi_largeur, 6)
        hors_zone = (proba <= bas) | (proba >= haut)
        if hors_zone.any() and (y[hors_zone] == predictions[hors_zone]).mean() >= cible:
            return {
                "cible_exactitude_hors_zone": cible,
                "seuil_bas": bas,
                "seuil_haut": haut,
                "demi_largeur": demi_largeur,
                "part_dans_la_zone": float(1 - hors_zone.mean()),
                "exactitude_hors_zone": float((y[hors_zone] == predictions[hors_zone]).mean()),
                "exactitude_dans_la_zone": (
                    float((y[~hors_zone] == predictions[~hors_zone]).mean()) if (~hors_zone).any() else None
                ),
            }
    raise ValueError("Aucune zone ne permet d'atteindre la cible d'exactitude.")


def appliquer_zone(y, proba, seuil_bas: float, seuil_haut: float) -> dict:
    """Mesure une zone déjà fixée : part des critiques incertaines et exactitude de chaque côté."""
    y, proba = np.asarray(y), np.asarray(proba)
    predictions = (proba >= 0.5).astype(int)
    hors_zone = (proba <= seuil_bas) | (proba >= seuil_haut)
    return {
        "part_dans_la_zone": float(1 - hors_zone.mean()),
        "effectif_dans_la_zone": int((~hors_zone).sum()),
        "exactitude_hors_zone": float((y[hors_zone] == predictions[hors_zone]).mean()) if hors_zone.any() else None,
        "exactitude_dans_la_zone": (
            float((y[~hors_zone] == predictions[~hors_zone]).mean()) if (~hors_zone).any() else None
        ),
    }

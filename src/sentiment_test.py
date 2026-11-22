"""Phase 5 : évaluation unique sur le test, en deux temps séparés.

1. `predire` : vérifie que les choix sont figés dans un commit, puis calcule les prédictions
   sur le test SANS charger les étiquettes, et scelle le fichier par son empreinte.
2. `mesurer` : lit les étiquettes du test une seule fois et mesure les prédictions scellées.

Les deux étapes refusent de s'exécuter une seconde fois. Rien n'est réentraîné ici.

Exemple d'utilisation :
    python -m src.sentiment_test --etape predire
    python -m src.sentiment_test --etape mesurer
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import evaluation as ev
from src import features as f
from src import models as m
from src import sentiment_eval as se
from src import sentiment_grille as g
from src.data import charger_brut
from src.preprocessing import RACINE, pipeline_a, retirer_notes

CHEMIN_SCEAU = RACINE / "reports" / "phase05_ouverture_test.json"
CHEMIN_PREDICTIONS = g.DOSSIER / "predictions_test.parquet"
CHEMIN_RESULTATS = g.DOSSIER / "resultats_test.json"
CHEMIN_ERREURS = g.DOSSIER / "erreurs_a_lire.parquet"
CHEMIN_QUASI = RACINE / "data" / "interim" / "phase3" / "quasi_doublons_train_test.parquet"
SEUIL_ALERTE_FUITE = 0.95
N_ERREURS_PAR_SENS = 10


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=RACINE, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def _sha256(chemin) -> str:
    return hashlib.sha256(chemin.read_bytes()).hexdigest().upper()


def verifier_gel() -> dict:
    """Refuse d'ouvrir le test si les choix ne sont pas dans un commit ou si un modèle a changé."""
    relatif = se.CHEMIN_CHOIX.relative_to(RACINE).as_posix()
    if not se.CHEMIN_CHOIX.exists():
        raise RuntimeError("Choix figés absents : lancer d'abord l'étape `gel`.")
    if _git("status", "--porcelain", "--", relatif):
        raise RuntimeError("Le fichier des choix n'est pas identique à sa version validée par un commit.")
    commit = _git("log", "-1", "--format=%H", "--", relatif)
    if not commit:
        raise RuntimeError("Le fichier des choix n'a jamais été validé par un commit.")
    choix = json.loads(se.CHEMIN_CHOIX.read_text(encoding="utf-8"))
    for famille, info in choix["modeles"].items():
        if _sha256(se.DOSSIER_MODELES / info["fichier"]) != info["sha256"]:
            raise RuntimeError(f"Le modèle {famille} a changé depuis le gel.")
    if _sha256(se.CHEMIN_OOF) != choix["sha256_probabilites_hors_pli"]:
        raise RuntimeError("Les probabilités hors pli ont changé depuis le gel.")
    id_train, id_test = f.charger_index()
    if f.empreinte_index(id_test) != choix["separation"]["empreinte_index_test"] or (
        f.empreinte_index(id_train) != choix["separation"]["empreinte_index_entrainement"]
    ):
        raise RuntimeError("La séparation a changé depuis le gel.")
    return {"choix": choix, "commit_gel": commit,
            "date_commit_gel": _git("log", "-1", "--format=%cI", commit)}


def _scores_vader(textes: list[str]) -> np.ndarray:
    return m.ReferenceVader().scores(textes)


def etape_predire() -> dict:
    """Prédit le test sans lire ses étiquettes, puis scelle le fichier de prédictions."""
    if CHEMIN_SCEAU.exists() or CHEMIN_PREDICTIONS.exists():
        raise RuntimeError("Les prédictions du test existent déjà : le test ne s'ouvre qu'une fois.")
    gel = verifier_gel()
    choix = gel["choix"]
    _, id_test = f.charger_index()
    brut = charger_brut()[["id_brut", "review"]].set_index("id_brut").loc[np.sort(id_test), "review"]
    ids = brut.index.to_numpy()
    bruts = brut.tolist()
    if choix["retirer_les_notes"]:
        textes = Parallel(n_jobs=g.N_PROCESSUS, batch_size=500)(delayed(pipeline_a)(retirer_notes(t)) for t in bruts)
    else:
        textes = Parallel(n_jobs=g.N_PROCESSUS, batch_size=500)(delayed(pipeline_a)(t) for t in bruts)
        cache = pd.read_parquet(
            f.CHEMIN_TEXTES, columns=["id_brut", "texte_a"], filters=[("partition", "==", f.PARTITION_TEST)]
        ).set_index("id_brut").loc[ids, "texte_a"].tolist()
        if textes != cache:
            raise RuntimeError("Le pipeline A recalculé sur le test diffère du cache.")
    textes = np.array(textes, dtype=object)

    colonnes = {"id_brut": ids}
    rechargement = {}
    for famille, info in choix["modeles"].items():
        chemin = se.DOSSIER_MODELES / info["fichier"]
        proba = m.charger_modele(chemin).predict_proba(textes)[:, 1]
        seconde = m.charger_modele(chemin).predict_proba(textes)[:, 1]
        rechargement[famille] = bool(np.array_equal(proba, seconde))
        colonnes[famille] = proba
    morceaux = [bruts[i:i + 1000] for i in range(0, len(bruts), 1000)]
    colonnes["vader_compound"] = np.concatenate(
        Parallel(n_jobs=g.N_PROCESSUS)(delayed(_scores_vader)(x) for x in morceaux)
    )
    pd.DataFrame(colonnes).to_parquet(CHEMIN_PREDICTIONS, index=False)
    sceau = {
        "commit_des_choix_figes": gel["commit_gel"],
        "date_commit_des_choix": gel["date_commit_gel"],
        "date_predictions": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sha256_predictions": _sha256(CHEMIN_PREDICTIONS),
        "critiques_de_test": int(len(ids)),
        "etiquettes_lues_pour_predire": False,
        "deux_rechargements_identiques": rechargement,
        "date_lecture_etiquettes": None,
        "nombre_de_lectures_des_etiquettes": 0,
    }
    CHEMIN_SCEAU.write_text(json.dumps(sceau, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sceau


def _mesurer_modele(y, score, predictions=None, probabilite: bool = True) -> dict:
    resultat = ev.metriques_classement(y, score, predictions=predictions)
    resultat["ic95"] = ev.bootstrap_ic(y, score, predictions=predictions, avec_brier=probabilite)
    if probabilite:
        resultat.update(ev.metriques_probabilite(y, score))
    return resultat


def etape_mesurer() -> dict:
    """Unique lecture des étiquettes du test. Aucune décision ne dépend de ce qui suit."""
    if not CHEMIN_SCEAU.exists():
        raise RuntimeError("Lancer d'abord l'étape `predire`.")
    sceau = json.loads(CHEMIN_SCEAU.read_text(encoding="utf-8"))
    if sceau["nombre_de_lectures_des_etiquettes"] != 0 or CHEMIN_RESULTATS.exists():
        raise RuntimeError("Les étiquettes du test ont déjà été lues : mesure refusée.")
    if _sha256(CHEMIN_PREDICTIONS) != sceau["sha256_predictions"]:
        raise RuntimeError("Le fichier de prédictions a changé depuis son scellement.")
    gel = verifier_gel()
    if gel["commit_gel"] != sceau["commit_des_choix_figes"]:
        raise RuntimeError("Les choix figés ont changé depuis les prédictions.")
    choix = gel["choix"]

    # Le compteur est écrit AVANT la lecture : une erreur ne permet pas de relire en silence.
    sceau["nombre_de_lectures_des_etiquettes"] = 1
    sceau["date_lecture_etiquettes"] = datetime.now().astimezone().isoformat(timespec="seconds")
    CHEMIN_SCEAU.write_text(json.dumps(sceau, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    predictions = pd.read_parquet(CHEMIN_PREDICTIONS)
    etiquettes = pd.read_parquet(
        f.CHEMIN_TEXTES, columns=["id_brut", "sentiment"], filters=[("partition", "==", f.PARTITION_TEST)]
    ).set_index("id_brut")
    y = (etiquettes.loc[predictions["id_brut"], "sentiment"] == "positive").astype(int).to_numpy()
    pd.DataFrame({"id_brut": predictions["id_brut"], "y": y}).to_parquet(g.DOSSIER / "etiquettes_test.parquet", index=False)

    modeles = {fam: _mesurer_modele(y, predictions[fam].to_numpy()) for fam in m.FAMILLES}
    compound = predictions["vader_compound"].to_numpy()
    modeles["vader"] = _mesurer_modele(y, compound, predictions=(compound >= 0).astype(int), probabilite=False)

    retenu = choix["modele_retenu"]["famille"]
    proba = predictions[retenu].to_numpy()
    zone = ev.appliquer_zone(y, proba, choix["zone_incertitude"]["seuil_bas"], choix["zone_incertitude"]["seuil_haut"])

    quasi = pd.read_parquet(CHEMIN_QUASI)
    ids_quasi = set(quasi["id_brut_test"].astype(int))
    sans_quasi = ~predictions["id_brut"].isin(ids_quasi).to_numpy()
    y_pred = (proba >= 0.5).astype(int)
    effet_quasi = {
        "critiques_retirees": int((~sans_quasi).sum()),
        "exactitude_avec": float((y_pred == y).mean()),
        "exactitude_sans": float((y_pred[sans_quasi] == y[sans_quasi]).mean()),
        "erreurs_parmi_les_quasi_doublons": int((y_pred[~sans_quasi] != y[~sans_quasi]).sum()),
    }
    effet_quasi["ecart_points"] = 100 * (effet_quasi["exactitude_avec"] - effet_quasi["exactitude_sans"])

    # Tirage des 20 erreurs à lire, fixé par la graine et indépendant de leur contenu.
    tirage = np.random.default_rng(se.g.GRAINE)
    faux_positifs = np.flatnonzero((y_pred == 1) & (y == 0))
    faux_negatifs = np.flatnonzero((y_pred == 0) & (y == 1))
    indices = np.concatenate([
        tirage.choice(faux_positifs, N_ERREURS_PAR_SENS, replace=False),
        tirage.choice(faux_negatifs, N_ERREURS_PAR_SENS, replace=False),
    ])
    brut = charger_brut(verifier=False).set_index("id_brut")
    ids_erreurs = predictions["id_brut"].to_numpy()[indices]
    pd.DataFrame({
        "id_brut": ids_erreurs, "y": y[indices], "proba_positive": proba[indices],
        "review": brut.loc[ids_erreurs, "review"].to_numpy(),
    }).to_parquet(CHEMIN_ERREURS, index=False)

    alerte = {fam: modeles[fam]["exactitude"] > SEUIL_ALERTE_FUITE
              for fam in ("regression_logistique", "svm_lineaire")}
    resultats = {
        "modeles": modeles,
        "modele_retenu": choix["modele_retenu"],
        "fiabilite_modele_retenu": ev.courbe_fiabilite(y, proba),
        "zone_incertitude": {**choix["zone_incertitude"], **zone},
        "quasi_doublons": effet_quasi,
        "erreurs": {"faux_positifs": int(len(faux_positifs)), "faux_negatifs": int(len(faux_negatifs))},
        "alerte_exactitude_lineaire_sup_95": alerte,
    }
    CHEMIN_RESULTATS.write_text(json.dumps(resultats, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultats


if __name__ == "__main__":
    analyseur = argparse.ArgumentParser(description="Évaluation unique sur le test.")
    analyseur.add_argument("--etape", choices=["predire", "mesurer"], required=True)
    etape = analyseur.parse_args().etape
    sortie = etape_predire() if etape == "predire" else etape_mesurer()
    print(json.dumps(sortie, ensure_ascii=False, indent=1)[:7000])

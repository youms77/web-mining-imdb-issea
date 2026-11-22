"""Phase 5, avant le test : choix des réglages, notes écrites, calibration, zone, gel des choix.

Toutes les décisions se prennent sur l'entraînement, en validation croisée, avec les règles
ci-dessous, écrites avant de lire les résultats. Ce module ne charge ni texte ni étiquette du test.

Exemple d'utilisation :
    python -m src.sentiment_eval --etape selection
    python -m src.sentiment_eval --etape vader
    python -m src.sentiment_eval --etape notes
    python -m src.sentiment_eval --etape calibration
    python -m src.sentiment_eval --etape gel
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score

from src import evaluation as ev
from src import features as f
from src import models as m
from src import sentiment_grille as g
from src.data import charger_brut
from src.features_eval import decider
from src.preprocessing import RACINE, pipeline_a, retirer_notes

DOSSIER = g.DOSSIER
DOSSIER_MODELES = RACINE / "models"
CHEMIN_CHOIX = RACINE / "reports" / "phase05_choix_figes.json"
CHEMIN_OOF = DOSSIER / "probabilites_hors_pli.parquet"

# --- Règles fixées avant de lire les résultats ------------------------------------------------
TOLERANCE_EXACTITUDE = 0.0005  # 0,05 point, même seuil qu'aux phases 2 et 3
CIBLE_HORS_ZONE = 0.95  # exactitude visée sur les critiques jugées assez sûres
METHODES = ("sigmoid", "isotonic")
REGLES = {
    "hyperparametres": (
        "Par famille : meilleure exactitude moyenne sur 5 plis. Parmi les réglages à moins de 0,05 point "
        "du meilleur, on garde le plus petit vocabulaire, puis le plus régularisé."
    ),
    "notes_ecrites": (
        "Les notes sont retirées si, pour au moins une famille, les garder apporte un gain d'au moins "
        "0,05 point et supérieur à 2 erreurs-types : ce gain viendrait d'une étiquette recopiée dans le texte. "
        "Sinon, le pipeline A reste inchangé."
    ),
    "calibration": (
        "Pour chaque famille, méthode au Brier moyen le plus bas (sigmoïde ou isotonique). "
        "Modèle retenu : Brier le plus bas parmi les versions calibrées. Si l'écart apparié au meilleur "
        "est inférieur à 2 erreurs-types, on préfère la meilleure exactitude, puis la sigmoïde."
    ),
    "zone_incertitude": (
        "Plus petite zone symétrique autour de 0,5 telle que l'exactitude hors zone atteigne 95 %, "
        "sur les probabilités hors pli du modèle retenu."
    ),
}


def _ecrire(nom: str, contenu) -> None:
    temporaire = DOSSIER / f"{nom}.tmp"
    temporaire.write_text(json.dumps(contenu, ensure_ascii=False, indent=1), encoding="utf-8")
    temporaire.replace(DOSSIER / nom)


def _lire(nom: str):
    return json.loads((DOSSIER / nom).read_text(encoding="utf-8"))


# --- 1. Hyperparamètres ---------------------------------------------------------------------------

def synthese_grille() -> pd.DataFrame:
    """Moyenne sur 5 plis de chaque réglage de la grille."""
    fichiers = sorted(g.DOSSIER_GRILLE.glob("pli*.json"))
    attendus = g.N_PLIS * len(m.GRILLE_MIN_DF) * len(m.GRILLE_MAX_DF)
    if len(fichiers) != attendus:
        raise ValueError(f"Grille incomplète : {len(fichiers)} fichiers sur {attendus}.")
    lignes = [l for x in fichiers for l in json.loads(x.read_text(encoding="utf-8"))["lignes"]]
    table = pd.DataFrame(lignes)
    cles = ["famille", "parametre", "valeur", "min_df", "max_df"]
    resume = table.sort_values("pli").groupby(cles, as_index=False).agg(
        exactitude=("exactitude", "mean"), f1=("f1", "mean"), roc_auc=("roc_auc", "mean"),
        termes=("termes", "mean"), alertes=("alertes_convergence", "sum"),
        plis=("pli", "count"), exactitude_par_pli=("exactitude", list),
    )
    if (resume["plis"] != g.N_PLIS).any():
        raise ValueError("Un réglage n'a pas ses 5 plis.")
    return resume


def choisir_reglage(resume: pd.DataFrame, famille: str) -> dict:
    """Applique la règle : meilleure exactitude, puis vocabulaire le plus petit, puis régularisation."""
    lignes = resume[resume["famille"] == famille].copy()
    classifieur = m.FAMILLES[famille][0]
    valeurs = m.GRILLE_CLASSIFIEUR[classifieur][1]
    lignes["rang_regularisation"] = lignes["valeur"].map(valeurs.index)
    meilleure = lignes["exactitude"].max()
    proches = lignes[lignes["exactitude"] >= meilleure - TOLERANCE_EXACTITUDE]
    choix = proches.sort_values(["termes", "rang_regularisation"]).iloc[0]
    premier = lignes.sort_values("exactitude", ascending=False).iloc[0]
    return {
        "famille": famille,
        "parametre": choix["parametre"],
        "valeur": float(choix["valeur"]),
        "min_df": int(choix["min_df"]),
        "max_df": float(choix["max_df"]),
        "exactitude_cv": float(choix["exactitude"]),
        "f1_cv": float(choix["f1"]),
        "roc_auc_cv": float(choix["roc_auc"]),
        "exactitude_par_pli": [float(v) for v in choix["exactitude_par_pli"]],
        "termes_moyens": int(choix["termes"]),
        "alertes_convergence": int(choix["alertes"]),
        "meilleure_exactitude_de_la_grille": float(meilleure),
        "reglage_de_la_meilleure_exactitude": {
            premier["parametre"]: float(premier["valeur"]),
            "min_df": int(premier["min_df"]), "max_df": float(premier["max_df"]),
        },
        "reglages_a_moins_de_0_05_point": int(len(proches)),
        "valeur_au_bord_de_la_grille": bool(choix["valeur"] in (valeurs[0], valeurs[-1])),
    }


def etape_selection() -> dict:
    resume = synthese_grille()
    choix = {famille: choisir_reglage(resume, famille) for famille in m.FAMILLES}
    resultat = {
        "regle": REGLES["hyperparametres"],
        "configurations_evaluees": int(len(resume)),
        "ajustements": int(len(resume) * g.N_PLIS),
        "alertes_convergence_totales": int(resume["alertes"].sum()),
        "choix": choix,
    }
    _ecrire("selection.json", resultat)
    resume.drop(columns="exactitude_par_pli").to_csv(DOSSIER / "grille_resume.csv", index=False)
    return resultat


# --- 2. Référence VADER (sans apprentissage) ------------------------------------------------------

def textes_bruts_entrainement(ids: np.ndarray) -> pd.Series:
    """Critiques brutes de l'entraînement seul, dans l'ordre de `ids`. La colonne d'étiquette est écartée."""
    brut = charger_brut()[["id_brut", "review"]].set_index("id_brut")
    return brut.loc[ids, "review"]


def _scores_vader(textes: list[str]) -> np.ndarray:
    return m.ReferenceVader().scores(textes)


def etape_vader() -> dict:
    ids, _, y = g.lire_entrainement()
    textes = textes_bruts_entrainement(ids).tolist()
    debut = time.perf_counter()
    _scores_vader(textes[:2000])
    sur_2000 = time.perf_counter() - debut
    print(f"VADER sur 2 000 critiques : {sur_2000:.1f} s", flush=True)
    debut = time.perf_counter()
    morceaux = [textes[i:i + 2000] for i in range(0, len(textes), 2000)]
    scores = np.concatenate(Parallel(n_jobs=g.N_PROCESSUS)(delayed(_scores_vader)(x) for x in morceaux))
    duree = time.perf_counter() - debut
    pd.DataFrame({"id_brut": ids, "compound": scores}).to_parquet(DOSSIER / "vader_train.parquet", index=False)
    predictions = (scores >= m.ReferenceVader.seuil).astype(int)
    resultat = {
        "perimetre": "entraînement complet : VADER n'apprend rien, seuil fixé à 0 sur compound",
        "chrono_2000_s": round(sur_2000, 1),
        "duree_s": round(duree, 1),
        **ev.metriques_classement(y, scores, predictions=predictions),
    }
    _ecrire("vader_train.json", resultat)
    return resultat


# --- 3. Notes écrites dans le texte --------------------------------------------------------------

def _preparer_textes_sans_notes(ids: np.ndarray, textes_a: np.ndarray) -> np.ndarray:
    chemin = DOSSIER / "textes_sans_notes.parquet"
    if chemin.exists():
        return pd.read_parquet(chemin)["texte"].to_numpy(dtype=object)
    bruts = textes_bruts_entrainement(ids).tolist()
    avec = Parallel(n_jobs=g.N_PROCESSUS, batch_size=500)(delayed(pipeline_a)(t) for t in bruts)
    if not np.array_equal(np.array(avec, dtype=object), textes_a):
        raise ValueError("Le pipeline A recalculé diffère du cache.")
    sans = Parallel(n_jobs=g.N_PROCESSUS, batch_size=500)(delayed(pipeline_a)(retirer_notes(t)) for t in bruts)
    pd.DataFrame({"id_brut": ids, "texte": sans}).to_parquet(chemin, index=False)
    return np.array(sans, dtype=object)


def _predire_pli(dossier, k: int, famille: str, choix: dict, y_fit) -> np.ndarray:
    x_fit = sparse.load_npz(dossier / f"pli{k}_fit.npz")
    x_val = sparse.load_npz(dossier / f"pli{k}_val.npz")
    modele = m.creer_pipeline_comptages(
        famille, {choix["parametre"]: choix["valeur"]}, min_df=choix["min_df"], max_df=choix["max_df"]
    )
    return modele.fit(x_fit, y_fit).predict(x_val)


def etape_notes() -> dict:
    ids, textes, y = g.lire_entrainement()
    sans_notes = _preparer_textes_sans_notes(ids, textes)
    modifiees = sans_notes != textes
    dossier_sans = DOSSIER / "plis_sans_notes"
    g.preparer_plis(sans_notes, y, dossier=dossier_sans)
    plis = g.plis_externes(textes, y)
    choix = _lire("selection.json")["choix"]
    travaux = [(d, k, fam) for d in (g.DOSSIER_PLIS, dossier_sans) for k in range(g.N_PLIS) for fam in m.FAMILLES]
    sorties = Parallel(n_jobs=g.N_PROCESSUS)(
        delayed(_predire_pli)(d, k, fam, choix[fam], y[plis[k][0]]) for d, k, fam in travaux
    )
    predictions = {(d, k, fam): p for (d, k, fam), p in zip(travaux, sorties)}
    familles = {}
    for fam in m.FAMILLES:
        par_pli = {"avec": [], "sans": []}
        vus = {"y": [], "avec": [], "sans": []}
        for k in range(g.N_PLIS):
            val = plis[k][1]
            avec, sans = predictions[(g.DOSSIER_PLIS, k, fam)], predictions[(dossier_sans, k, fam)]
            par_pli["avec"].append(float(accuracy_score(y[val], avec)))
            par_pli["sans"].append(float(accuracy_score(y[val], sans)))
            masque = modifiees[val]
            vus["y"].append(y[val][masque])
            vus["avec"].append(avec[masque])
            vus["sans"].append(sans[masque])
        y_notes = np.concatenate(vus["y"])
        familles[fam] = {
            "exactitude_avec_notes": float(np.mean(par_pli["avec"])),
            "exactitude_sans_notes": float(np.mean(par_pli["sans"])),
            "exactitude_par_pli_avec": par_pli["avec"],
            "exactitude_par_pli_sans": par_pli["sans"],
            "effet_de_garder_les_notes": decider(
                {"exactitude_par_pli": par_pli["sans"]}, {"exactitude_par_pli": par_pli["avec"]}
            ),
            "sur_les_critiques_contenant_une_note": {
                "effectif": int(len(y_notes)),
                "exactitude_avec_notes": float((np.concatenate(vus["avec"]) == y_notes).mean()),
                "exactitude_sans_notes": float((np.concatenate(vus["sans"]) == y_notes).mean()),
            },
        }
    retirer = any(v["effet_de_garder_les_notes"]["ameliore"] for v in familles.values())
    resultat = {
        "regle": REGLES["notes_ecrites"],
        "critiques_modifiees_par_le_retrait": int(modifiees.sum()),
        "part_de_l_entrainement_pct": round(100 * float(modifiees.mean()), 2),
        "familles": familles,
        "retirer_les_notes": bool(retirer),
    }
    _ecrire("notes.json", resultat)
    return resultat


# --- 4. Calibration, en validation croisée imbriquée ---------------------------------------------

def _job_calibration(dossier, k: int, famille: str, choix: dict, y_fit) -> dict:
    """Pli externe k : calibration ajustée sur la partie d'ajustement, mesurée sur le pli de validation."""
    x_fit = sparse.load_npz(dossier / f"pli{k}_fit.npz")
    x_val = sparse.load_npz(dossier / f"pli{k}_val.npz")
    base = m.creer_pipeline_comptages(
        famille, {choix["parametre"]: choix["valeur"]}, min_df=choix["min_df"], max_df=choix["max_df"]
    )
    sorties = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if hasattr(base.named_steps["classifieur"], "predict_proba"):
            sorties["brut"] = clone(base).fit(x_fit, y_fit).predict_proba(x_val)[:, 1]
        for methode in METHODES:
            sorties[methode] = m.calibrer(clone(base), methode).fit(x_fit, y_fit).predict_proba(x_val)[:, 1]
    return sorties


def etape_calibration() -> dict:
    ids, textes, y = g.lire_entrainement()
    notes = _lire("notes.json")
    if notes["retirer_les_notes"]:
        dossier = DOSSIER / "plis_sans_notes"
    else:
        dossier = g.DOSSIER_PLIS
    plis = g.plis_externes(textes, y)
    choix = _lire("selection.json")["choix"]
    debut = time.perf_counter()
    travaux = [(k, fam) for k in range(g.N_PLIS) for fam in m.FAMILLES]
    sorties = Parallel(n_jobs=g.N_PROCESSUS, verbose=5)(
        delayed(_job_calibration)(dossier, k, fam, choix[fam], y[plis[k][0]]) for k, fam in travaux
    )
    duree = time.perf_counter() - debut

    oof: dict[str, np.ndarray] = {}
    par_pli: dict[str, dict[str, list]] = {}
    for (k, fam), sortie in zip(travaux, sorties):
        val = plis[k][1]
        for methode, proba in sortie.items():
            cle = f"{fam}__{methode}"
            oof.setdefault(cle, np.full(len(y), np.nan))[val] = proba
            mesures = par_pli.setdefault(cle, {"brier": [], "log_loss": [], "ece": [], "exactitude": []})
            p = ev.metriques_probabilite(y[val], proba)
            for nom in ("brier", "log_loss", "ece"):
                mesures[nom].append(p[nom])
            mesures["exactitude"].append(float(accuracy_score(y[val], (proba >= 0.5).astype(int))))
    tableau = pd.DataFrame({"id_brut": ids, "y": y, **oof})
    if tableau.isna().any().any():
        raise ValueError("Probabilités hors pli incomplètes.")
    tableau.to_parquet(CHEMIN_OOF, index=False)

    candidats = {}
    for cle, mesures in par_pli.items():
        fam, methode = cle.split("__")
        candidats[cle] = {
            "famille": fam, "methode": methode,
            **{f"{nom}_moyen": float(np.mean(v)) for nom, v in mesures.items()},
            **{f"{nom}_par_pli": v for nom, v in mesures.items()},
            "ensemble_hors_pli": ev.metriques_probabilite(y, oof[cle]),
            "fiabilite": ev.courbe_fiabilite(y, oof[cle]),
        }
    methode_par_famille = {
        fam: min(METHODES, key=lambda x: candidats[f"{fam}__{x}"]["brier_moyen"]) for fam in m.FAMILLES
    }
    calibres = {fam: candidats[f"{fam}__{meth}"] for fam, meth in methode_par_famille.items()}
    meilleure = min(calibres, key=lambda x: calibres[x]["brier_moyen"])
    equivalents = {}
    for fam, c in calibres.items():
        ecart = ev.ecart_apparie(c["brier_par_pli"], calibres[meilleure]["brier_par_pli"])
        equivalents[fam] = {**ecart, "equivalent_au_meilleur": bool(fam == meilleure or ecart["ecart"] < 2 * ecart["erreur_type"])}
    admissibles = [fam for fam, e in equivalents.items() if e["equivalent_au_meilleur"]]
    retenu = max(
        admissibles,
        key=lambda x: (round(calibres[x]["exactitude_moyen"], 6), methode_par_famille[x] == "sigmoid"),
    )
    resultat = {
        "regle": REGLES["calibration"],
        "protocole": (
            "Validation croisée imbriquée : 5 plis externes, et dans chacun 5 plis internes pour le calibrateur. "
            "Les probabilités sont mesurées sur le pli externe, jamais vu par le modèle ni par le calibrateur."
        ),
        "duree_min": round(duree / 60, 1),
        "candidats": candidats,
        "methode_par_famille": methode_par_famille,
        "plus_bas_brier": meilleure,
        "ecarts_de_brier_au_meilleur": equivalents,
        "famille_retenue": retenu,
        "methode_retenue": methode_par_famille[retenu],
    }
    _ecrire("calibration.json", resultat)
    return resultat


# --- 5. Gel : modèles finaux et choix écrits avant le test ---------------------------------------

def _ajuster_final(famille: str, choix: dict, methode: str, textes, y) -> dict:
    pipeline = m.creer_pipeline_texte(
        famille, {choix["parametre"]: choix["valeur"]}, min_df=choix["min_df"], max_df=choix["max_df"]
    )
    debut = time.perf_counter()
    modele = m.calibrer(pipeline, methode).fit(textes, y)
    duree = time.perf_counter() - debut
    info = m.sauvegarder_modele(modele, DOSSIER_MODELES / f"sentiment_{famille}.joblib")
    controle = textes[:2000]
    relu = m.charger_modele(DOSSIER_MODELES / f"sentiment_{famille}.joblib")
    identique = bool(np.array_equal(modele.predict_proba(controle), relu.predict_proba(controle)))
    vocabulaire = modele.calibrated_classifiers_[0].estimator.named_steps["vectoriseur"].vocabulary_
    return {**info, "duree_ajustement_s": round(duree, 1), "termes": len(vocabulaire),
            "rechargement_identique_sur_2000_critiques_d_entrainement": identique}


def etape_gel() -> dict:
    ids, textes_a, y = g.lire_entrainement()
    notes = _lire("notes.json")
    textes = _preparer_textes_sans_notes(ids, textes_a) if notes["retirer_les_notes"] else textes_a
    selection, calibration = _lire("selection.json"), _lire("calibration.json")
    methodes = calibration["methode_par_famille"]
    fichiers = Parallel(n_jobs=3)(
        delayed(_ajuster_final)(fam, selection["choix"][fam], methodes[fam], textes, y) for fam in m.FAMILLES
    )
    modeles = dict(zip(m.FAMILLES, fichiers))
    retenu = calibration["famille_retenue"]
    methode = calibration["methode_retenue"]
    oof = pd.read_parquet(CHEMIN_OOF)
    proba = oof[f"{retenu}__{methode}"].to_numpy()
    zone = ev.choisir_zone_incertitude(y, proba, cible=CIBLE_HORS_ZONE)
    plis = g.plis_externes(textes_a, y)
    stabilite = [ev.appliquer_zone(y[v], proba[v], zone["seuil_bas"], zone["seuil_haut"]) for _, v in plis]
    id_train, id_test = f.charger_index()
    choix = {
        "date_gel": datetime.now().astimezone().isoformat(timespec="seconds"),
        "statut": "figé avant toute lecture du test",
        "regles": REGLES,
        "separation": {
            "entrainement": int(len(id_train)), "test": int(len(id_test)),
            "empreinte_index_entrainement": f.empreinte_index(id_train),
            "empreinte_index_test": f.empreinte_index(id_test),
        },
        "retirer_les_notes": notes["retirer_les_notes"],
        "pretraitement_des_modeles": (
            "src.models.preparer_texte : retirer_notes puis pipeline_a"
            if notes["retirer_les_notes"] else "pipeline_a"
        ),
        "reglages_par_famille": {
            fam: {k: selection["choix"][fam][k] for k in ("parametre", "valeur", "min_df", "max_df")}
            | {"methode_calibration": methodes[fam]}
            for fam in m.FAMILLES
        },
        "modele_retenu": {"famille": retenu, "methode_calibration": methode},
        "zone_incertitude": {k: zone[k] for k in ("cible_exactitude_hors_zone", "seuil_bas", "seuil_haut")},
        "zone_mesuree_hors_pli": zone,
        "zone_stabilite_par_pli": stabilite,
        "modeles": modeles,
        "sha256_probabilites_hors_pli": hashlib.sha256(CHEMIN_OOF.read_bytes()).hexdigest().upper(),
    }
    CHEMIN_CHOIX.write_text(json.dumps(choix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return choix


ETAPES = {
    "selection": etape_selection, "vader": etape_vader, "notes": etape_notes,
    "calibration": etape_calibration, "gel": etape_gel,
}

if __name__ == "__main__":
    analyseur = argparse.ArgumentParser(description="Phase 5 : décisions prises avant le test.")
    analyseur.add_argument("--etape", choices=list(ETAPES), required=True)
    sortie = ETAPES[analyseur.parse_args().etape]()
    print(json.dumps(sortie, ensure_ascii=False, indent=1)[:6000])

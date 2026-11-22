"""Phase 5, après l'unique lecture du test : coefficients, erreurs, figures et clé `sentiment`.

Ce module ne réentraîne rien et ne change aucun choix. Il relit les résultats déjà scellés.

Exemple d'utilisation :
    python -m src.sentiment_bilan
"""
from __future__ import annotations

import collections
import json
import re

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from src import exploration as ex
from src import models as m
from src import sentiment_eval as se
from src import sentiment_grille as g
from src import sentiment_test as st
from src.preprocessing import RACINE

CHEMIN_ANNOTATIONS = RACINE / "src" / "ressources" / "erreurs_annotees.json"
COULEUR_RETENU, COULEUR_NEUTRE, COULEUR_BRUT = "#6250d6", "#8a897f", "#eda100"
SOURCE = "Source : IMDB Dataset (Kaggle)"
DUREE_GRILLE_INITIALE_MIN = 4.7  # mesurée avant l'extension de la grille (144 réglages)


def _lire_json(chemin):
    return json.loads(chemin.read_text(encoding="utf-8"))


def _pct(valeur: float, decimales: int = 2) -> str:
    return f"{ex.fr(100 * valeur, decimales)} %"


# --- Interprétation et erreurs -----------------------------------------------------------------

def coefficients(n: int = 20) -> dict:
    """Termes les plus influents de la régression logistique, et recoupement avec le SVM retenu."""
    def extraire(famille: str):
        modele = m.charger_modele(se.DOSSIER_MODELES / f"sentiment_{famille}.joblib")
        pipeline = modele.calibrated_classifiers_[0].estimator
        noms = pipeline.named_steps["vectoriseur"].get_feature_names_out()
        return noms, pipeline.named_steps["classifieur"].coef_.ravel()

    noms, poids = extraire("regression_logistique")
    ordre = np.argsort(poids)
    noms_svm, poids_svm = extraire("svm_lineaire")
    ordre_svm = np.argsort(poids_svm)
    return {
        "modele": "régression logistique gelée (C = 30), apprise sur tout l'entraînement",
        "lecture": "Un coefficient positif pousse vers « positif » à valeurs égales des autres termes. Ce n'est pas un effet causal.",
        "termes": int(len(noms)),
        "vers_positif": [{"terme": str(noms[i]), "coefficient": round(float(poids[i]), 3)} for i in ordre[::-1][:n]],
        "vers_negatif": [{"terme": str(noms[i]), "coefficient": round(float(poids[i]), 3)} for i in ordre[:n]],
        "termes_communs_avec_le_svm_sur_20": {
            "positif": len(set(noms[ordre[::-1][:n]]) & set(noms_svm[ordre_svm[::-1][:n]])),
            "negatif": len(set(noms[ordre[:n]]) & set(noms_svm[ordre_svm[:n]])),
        },
    }


def erreurs_annotees(resultats: dict) -> dict:
    """Associe chaque erreur lue à sa cause. Les textes des critiques ne sont pas recopiés."""
    annotations = _lire_json(CHEMIN_ANNOTATIONS)
    tirees = pd.read_parquet(st.CHEMIN_ERREURS).set_index("id_brut")
    ids = [e["id_brut"] for e in annotations["erreurs"]]
    if set(ids) != set(tirees.index) or len(ids) != len(set(ids)):
        raise ValueError("Les annotations ne correspondent pas aux 20 erreurs tirées.")
    lignes = []
    for e in annotations["erreurs"]:
        t = tirees.loc[e["id_brut"]]
        if (t["proba_positive"] >= 0.5) == bool(t["y"]):
            raise ValueError(f"La critique {e['id_brut']} n'est pas une erreur du modèle.")
        lignes.append({
            "id_brut": int(e["id_brut"]),
            "classe_reelle": "positif" if t["y"] else "négatif",
            "probabilite_positive": round(float(t["proba_positive"]), 3),
            "dans_la_zone_incertaine": bool(
                resultats["zone_incertitude"]["seuil_bas"] < t["proba_positive"] < resultats["zone_incertitude"]["seuil_haut"]
            ),
            "cause": annotations["causes"][e["cause"]],
            "explication": e["explication"],
        })
    compte = collections.Counter(l["cause"] for l in lignes)
    return {
        "methode": annotations["methode"],
        "erreurs_totales_du_modele_sur_le_test": resultats["erreurs"]["faux_positifs"] + resultats["erreurs"]["faux_negatifs"],
        "faux_positifs": resultats["erreurs"]["faux_positifs"],
        "faux_negatifs": resultats["erreurs"]["faux_negatifs"],
        "repartition_des_20": dict(compte.most_common()),
        "erreurs_lues_dans_la_zone_incertaine": sum(l["dans_la_zone_incertaine"] for l in lignes),
        "avertissement": "20 critiques sur l'ensemble des erreurs : cet échantillon illustre les causes, il n'en mesure pas la fréquence.",
        "lignes": lignes,
    }


# --- Figures : texte de 16 pt minimum ----------------------------------------------------------

def figure_comparaison(resultats: dict) -> None:
    ex.appliquer_style()
    modeles = sorted(resultats["modeles"], key=lambda x: resultats["modeles"][x]["exactitude"])
    retenu = resultats["modele_retenu"]["famille"]
    fig, ax = plt.subplots(figsize=(14, 6.8))
    for y, nom in enumerate(modeles):
        r = resultats["modeles"][nom]
        v, b, h = (100 * x for x in (r["exactitude"], r["ic95"]["exactitude"]["bas"], r["ic95"]["exactitude"]["haut"]))
        couleur = COULEUR_RETENU if nom == retenu else COULEUR_NEUTRE
        ax.plot([b, h], [y, y], color=couleur, linewidth=3, solid_capstyle="round", zorder=2)
        ax.scatter(v, y, s=150, color=couleur, edgecolor=ex.SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{ex.fr(v, 2)} %  [{ex.fr(b, 2)} ; {ex.fr(h, 2)}]", (h, y), xytext=(14, 0),
                    textcoords="offset points", va="center", fontsize=16,
                    fontweight="bold" if nom == retenu else "normal", color=ex.ENCRE)
    ax.set_yticks(range(len(modeles)), [
        m.LIBELLES_FAMILLES[n] + (" (retenu)" if n == retenu else "") for n in modeles
    ])
    ax.set_xlim(66, 104)
    ax.set_xticks([70, 75, 80, 85, 90, 95])
    ax.grid(axis="x")
    ax.set_xlabel("Exactitude sur le test (%), axe tronqué")
    ax.set_title("Les modèles supervisés dépassent nettement le lexique", loc="left", fontweight="bold")
    fig.text(0.01, -0.03, f"Test unique, 9 915 critiques. Barres : IC à 95 % par bootstrap (2 000 tirages). {SOURCE}.",
             fontsize=16, color=ex.ENCRE_2)
    fig.tight_layout()
    ex._sauver(fig, "12_comparaison_modeles.png")


def _tracer_fiabilite(ax, courbe, couleur, libelle, marqueur) -> None:
    x = [l["probabilite_moyenne"] for l in courbe]
    y = [l["frequence_observee"] for l in courbe]
    ax.plot(x, y, color=couleur, linewidth=2.5, marker=marqueur, markersize=9, markeredgecolor=ex.SURFACE,
            markeredgewidth=1.5, label=libelle)


def figure_fiabilite(calibration: dict, resultats: dict) -> None:
    ex.appliquer_style()
    fig, (gauche, droite) = plt.subplots(1, 2, figsize=(16, 7.6), sharey=True)
    candidats = calibration["candidats"]
    for ax in (gauche, droite):
        ax.plot([0, 1], [0, 1], color=ex.AXE, linewidth=2, linestyle="--", label="Calibration parfaite")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: ex.fr(v, 1)))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: ex.fr(v, 1)))
        ax.set_xlabel("Probabilité annoncée")
    methode_nb = calibration["methode_par_famille"]["naive_bayes_bow"]
    _tracer_fiabilite(gauche, candidats["naive_bayes_bow__brut"]["fiabilite"], COULEUR_BRUT, "Brut", "o")
    _tracer_fiabilite(gauche, candidats[f"naive_bayes_bow__{methode_nb}"]["fiabilite"], COULEUR_RETENU, "Calibré (isotonique)", "s")
    gauche.set_ylabel("Part réelle d'avis positifs")
    gauche.set_title("Naive Bayes, BoW (validation)", loc="left", fontweight="bold")
    gauche.legend(frameon=False, loc="upper left")
    famille, methode = calibration["famille_retenue"], calibration["methode_retenue"]
    _tracer_fiabilite(droite, candidats[f"{famille}__{methode}"]["fiabilite"], COULEUR_RETENU, "Validation (hors pli)", "s")
    _tracer_fiabilite(droite, resultats["fiabilite_modele_retenu"], ex.COULEURS["positive"], "Test", "o")
    droite.set_title("SVM linéaire calibré (retenu)", loc="left", fontweight="bold")
    droite.legend(frameon=False, loc="upper left")
    fig.text(0.01, -0.02, f"Tranches de 0,1. Plus la courbe suit la diagonale, plus la probabilité est fiable. {SOURCE}.",
             fontsize=16, color=ex.ENCRE_2)
    fig.tight_layout()
    ex._sauver(fig, "13_fiabilite_calibration.png")


def figure_zone(choix: dict, resultats: dict) -> None:
    ex.appliquer_style()
    famille = choix["modele_retenu"]["famille"]
    methode = choix["modele_retenu"]["methode_calibration"]
    oof = pd.read_parquet(se.CHEMIN_OOF)
    bas, haut = choix["zone_incertitude"]["seuil_bas"], choix["zone_incertitude"]["seuil_haut"]
    fig, ax = plt.subplots(figsize=(14, 7))
    bornes = np.linspace(0, 1, 41)
    for classe, valeur in (("negative", 0), ("positive", 1)):
        ax.hist(oof.loc[oof["y"] == valeur, f"{famille}__{methode}"], bins=bornes, histtype="step", linewidth=2.5,
                color=ex.COULEURS[classe], label=f"Avis réellement {ex.LIBELLES[classe].lower()}s")
    ax.axvspan(bas, haut, color=ex.GRILLE, alpha=0.7, zorder=0)
    sommet = ax.get_ylim()[1]
    ax.set_ylim(0, sommet * 1.16)
    ax.text((bas + haut) / 2, sommet * 1.08, f"Zone incertaine : {ex.fr(bas, 2)} à {ex.fr(haut, 2)}", ha="center",
            va="center", fontsize=16, fontweight="bold", color=ex.ENCRE)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: ex.fr(v)))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: ex.fr(v, 1)))
    ax.set_xlabel("Probabilité d'avis positif (hors pli, entraînement)")
    ax.set_ylabel("Nombre de critiques")
    ax.set_title("Une zone pour les avis que le modèle ne sait pas trancher", loc="left", fontweight="bold")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.9))
    z = resultats["zone_incertitude"]
    fig.text(0.01, -0.07,
             f"Seuils fixés sur la validation. Sur le test : {_pct(z['part_dans_la_zone'], 1)} des critiques dans la zone, "
             f"exactitude {_pct(z['exactitude_dans_la_zone'], 1)} dedans\net {_pct(z['exactitude_hors_zone'], 1)} dehors. {SOURCE}.",
             fontsize=16, color=ex.ENCRE_2)
    fig.tight_layout()
    ex._sauver(fig, "14_zone_incertitude.png")


def figure_confusion(resultats: dict) -> None:
    ex.appliquer_style()
    c = resultats["modeles"][resultats["modele_retenu"]["famille"]]["matrice_confusion"]
    matrice = np.array([[c["vrais_negatifs"], c["faux_positifs"]], [c["faux_negatifs"], c["vrais_positifs"]]])
    parts = matrice / matrice.sum(axis=1, keepdims=True)
    rampe = LinearSegmentedColormap.from_list("rampe", ["#eeecfb", "#6250d6", "#2f2580"])
    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    ax.imshow(parts, cmap=rampe, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            couleur = "#ffffff" if parts[i, j] > 0.5 else ex.ENCRE
            ax.text(j, i - 0.09, ex.fr(matrice[i, j]), ha="center", va="center", fontsize=26, fontweight="bold", color=couleur)
            ax.text(j, i + 0.15, f"{_pct(parts[i, j], 1)} de la ligne", ha="center", va="center", fontsize=16, color=couleur)
    ax.set_xticks([0, 1], ["Négatif", "Positif"])
    ax.set_yticks([0, 1], ["Négatif", "Positif"])
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.tick_params(length=0)
    for cote in ax.spines.values():
        cote.set_visible(False)
    ax.set_title("Matrice de confusion du SVM calibré", loc="left", fontweight="bold")
    fig.text(0.02, -0.03, f"Test unique, 9 915 critiques, seuil 0,5. {SOURCE}.", fontsize=16, color=ex.ENCRE_2)
    fig.tight_layout()
    ex._sauver(fig, "15_matrice_confusion.png")


def figure_coefficients(coef: dict, n: int = 15) -> None:
    ex.appliquer_style()
    fig, (gauche, droite) = plt.subplots(1, 2, figsize=(16, 9.2))
    maximum = max(abs(x["coefficient"]) for x in coef["vers_positif"][:n] + coef["vers_negatif"][:n])
    for ax, cle, classe, titre in ((gauche, "vers_negatif", "negative", "Poussent vers « négatif »"),
                                   (droite, "vers_positif", "positive", "Poussent vers « positif »")):
        lignes = coef[cle][:n]
        y = np.arange(len(lignes))
        valeurs = [abs(x["coefficient"]) for x in lignes]
        ax.barh(y, valeurs, color=ex.COULEURS[classe], height=0.68)
        for position, valeur in zip(y, valeurs):
            ax.text(valeur + maximum * 0.02, position, ex.fr(valeur, 1), va="center", fontsize=16, color=ex.ENCRE)
        ax.set_yticks(y, [x["terme"] for x in lignes])
        ax.invert_yaxis()
        ax.set_xlim(0, maximum * 1.22)
        ax.set_xlabel("Poids absolu du coefficient")
        ax.set_title(titre, loc="left", fontweight="bold")
        ax.grid(axis="x")
    fig.text(0.01, -0.02, f"Régression logistique sur TF-IDF, entraînement. Association, pas causalité. {SOURCE}.",
             fontsize=16, color=ex.ENCRE_2)
    fig.tight_layout()
    ex._sauver(fig, "16_coefficients_regression_logistique.png")


# --- Orchestration -----------------------------------------------------------------------------

def duree_grille_etendue() -> float | None:
    chemin = g.DOSSIER / "grille.log"
    if not chemin.exists():
        return None
    for encodage in ("utf-16", "utf-8"):
        try:
            texte = chemin.read_text(encoding=encodage)
        except UnicodeError:
            continue
        trouve = re.search(r"Grille termin\S* en ([\d.]+) min", texte)
        if trouve:
            return float(trouve.group(1))
    return None


def executer() -> dict:
    """Assemble la clé `sentiment` de metrics.json, marquée comme gelée, et trace les figures."""
    sceau = _lire_json(st.CHEMIN_SCEAU)
    if sceau["nombre_de_lectures_des_etiquettes"] != 1:
        raise RuntimeError("Le test doit avoir été lu exactement une fois.")
    choix = _lire_json(se.CHEMIN_CHOIX)
    resultats = _lire_json(st.CHEMIN_RESULTATS)
    selection, calibration = _lire_json(g.DOSSIER / "selection.json"), _lire_json(g.DOSSIER / "calibration.json")
    notes, vader = _lire_json(g.DOSSIER / "notes.json"), _lire_json(g.DOSSIER / "vader_train.json")
    chrono = _lire_json(g.DOSSIER / "chrono.json")
    coef = coefficients()
    erreurs = erreurs_annotees(resultats)
    figure_comparaison(resultats)
    figure_fiabilite(calibration, resultats)
    figure_zone(choix, resultats)
    figure_confusion(resultats)
    figure_coefficients(coef)

    retenu = choix["modele_retenu"]["famille"]
    bilan = {
        "gel": True,
        "date_gel_des_choix": choix["date_gel"],
        "commit_des_choix_figes": sceau["commit_des_choix_figes"],
        "date_commit_des_choix": sceau["date_commit_des_choix"],
        "date_predictions_test": sceau["date_predictions"],
        "date_lecture_etiquettes_test": sceau["date_lecture_etiquettes"],
        "nombre_de_lectures_des_etiquettes_test": sceau["nombre_de_lectures_des_etiquettes"],
        "sha256_predictions_test": sceau["sha256_predictions"],
        "regles_fixees_avant_resultats": choix["regles"],
        "temps_de_calcul": {
            "chronometrage_grille_etendue": chrono,
            "grille_initiale_144_reglages_min": DUREE_GRILLE_INITIALE_MIN,
            "grille_etendue_300_reglages_min": duree_grille_etendue(),
            "calibration_imbriquee_min": calibration["duree_min"],
            "ajustements_finaux_s": {k: v["duree_ajustement_s"] for k, v in choix["modeles"].items()},
        },
        "recherche_hyperparametres": {
            **selection,
            "extension": (
                "La première grille s'arrêtait à C = 30 (régression) et C = 1 (SVM), valeurs choisies au bord, "
                "et min_df = 5 battait 10 à C élevé. La grille a été étendue sur l'entraînement avant toute lecture du test."
            ),
        },
        "reference_vader_entrainement": vader,
        "notes_ecrites": notes,
        "calibration": calibration,
        "choix_figes": {k: choix[k] for k in (
            "retirer_les_notes", "pretraitement_des_modeles", "reglages_par_famille", "modele_retenu",
            "zone_incertitude", "zone_mesuree_hors_pli", "zone_stabilite_par_pli", "modeles",
        )},
        "test": resultats,
        "resume_modele_retenu": {
            "modele": f"{m.LIBELLES_FAMILLES[retenu]}, calibration {choix['modele_retenu']['methode_calibration']}",
            **{k: resultats["modeles"][retenu][k] for k in (
                "exactitude", "precision", "rappel", "f1", "roc_auc", "brier", "log_loss", "ece", "ic95", "matrice_confusion"
            )},
        },
        "coefficients": coef,
        "analyse_erreurs": erreurs,
        "figures": [
            "reports/figures/12_comparaison_modeles.png", "reports/figures/13_fiabilite_calibration.png",
            "reports/figures/14_zone_incertitude.png", "reports/figures/15_matrice_confusion.png",
            "reports/figures/16_coefficients_regression_logistique.png",
        ],
        "limites": [
            "Les hyperparamètres ont été choisis avec les notes écrites présentes ; les notes ont ensuite été retirées pour la calibration et les modèles finaux.",
            "Le corpus n'indique pas le film : une même œuvre peut figurer dans l'entraînement et le test, le score peut donc être un peu optimiste.",
            "Les avis neutres (notes 5 et 6) sont absents du corpus : la zone d'incertitude n'a pas été validée sur de vrais avis neutres.",
            "Le classement des 20 erreurs est une interprétation sur un petit échantillon.",
        ],
    }
    ex.enregistrer_metrics("sentiment", bilan)
    return bilan


if __name__ == "__main__":
    sortie = executer()
    print(json.dumps({k: sortie[k] for k in ("gel", "commit_des_choix_figes", "resume_modele_retenu")}, ensure_ascii=False, indent=1))
    print(json.dumps(sortie["analyse_erreurs"]["repartition_des_20"], ensure_ascii=False))
    print("Erreurs lues dans la zone :", sortie["analyse_erreurs"]["erreurs_lues_dans_la_zone_incertaine"])
    print("Grille étendue (min) :", sortie["temps_de_calcul"]["grille_etendue_300_reglages_min"])
    print("Vers positif :", [x["terme"] for x in sortie["coefficients"]["vers_positif"][:12]])
    print("Vers négatif :", [x["terme"] for x in sortie["coefficients"]["vers_negatif"][:12]])
    print("Communs SVM :", sortie["coefficients"]["termes_communs_avec_le_svm_sur_20"])

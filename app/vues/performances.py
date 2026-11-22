"""Page « Performances » : comparaison des modèles, confusion et calibration sur le test."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import interface as ui
from src import application as ap
from src.models import LIBELLES_FAMILLES

artefacts = ui.charger()
test = artefacts.metriques["sentiment"]["test"]
retenu = test["modele_retenu"]["famille"]
resume = artefacts.metriques["sentiment"]["resume_modele_retenu"]
zone = test["zone_incertitude"]
modeles = test["modeles"]
n_test = sum(modeles[retenu]["matrice_confusion"].values())

ui.entete(
    f"{ap.fr_pct(resume['exactitude'], 2)} d'exactitude, et un modèle qui sait quand il hésite",
    f"Évaluation unique sur {ap.fr_entier(n_test)} critiques jamais vues pendant l'entraînement. Modèle retenu : {resume['modele'].replace('sigmoid', 'sigmoïde')}.",
)

a, b, c, d = st.columns(4)
a.metric("Exactitude", ap.fr_pct(resume["exactitude"], 2), border=True,
         help=f"IC à 95 % : {ap.fr_pct(resume['ic95']['exactitude']['bas'], 2)} à {ap.fr_pct(resume['ic95']['exactitude']['haut'], 2)}")
b.metric("Score F1", ap.fr_nombre(resume["f1"], 3), border=True)
c.metric("ROC-AUC", ap.fr_nombre(resume["roc_auc"], 3), border=True)
d.metric("Calibration (ECE)", ap.fr_nombre(resume["ece"], 4), border=True)

col_gauche, col_droite = st.columns([1.1, 1], gap="large")
with col_gauche:
    with ui.carte("comparaison"):
        st.subheader("Comparaison des modèles", anchor=False)
        lignes = [
            {"nom": LIBELLES_FAMILLES[f], "exactitude": modeles[f]["exactitude"], "bas": modeles[f]["ic95"]["exactitude"]["bas"],
             "haut": modeles[f]["ic95"]["exactitude"]["haut"], "retenu": f == retenu}
            for f in modeles
        ]
        ui.afficher(ui.points_modeles(lignes), "comparaison_modeles", "Exactitude de chaque modèle sur le test, avec son intervalle à 95 %.")
        st.caption("Point : exactitude. Trait : intervalle de confiance à 95 %. L'axe ne part pas de zéro.")
with col_droite:
    with ui.carte("confusion"):
        st.subheader("Matrice de confusion", anchor=False)
        famille = st.selectbox("Modèle", list(modeles), index=list(modeles).index(retenu), format_func=LIBELLES_FAMILLES.get, key="modele_confusion")
        ui.afficher(ui.matrice_confusion(modeles[famille]["matrice_confusion"]), "confusion", "Matrice de confusion du modèle choisi.")
        st.caption("Dans chaque case : effectif et part de la vraie classe.")

col_calib, col_zone = st.columns([1.1, 1], gap="large")
with col_calib:
    with ui.carte("calibration"):
        st.subheader("Une probabilité qui veut dire quelque chose", anchor=False)
        ui.afficher(ui.courbe_calibration(test["fiabilite_modele_retenu"]), "calibration", "Courbe de calibration du SVM sur le test.")
        st.caption("Plus la courbe suit la diagonale, plus la probabilité annoncée correspond à la fréquence observée.")
with col_zone:
    with ui.carte("zone"):
        st.subheader("La zone d'incertitude", anchor=False)
        st.markdown(
            f"Entre **{ap.fr_nombre(zone['seuil_bas'], 2)}** et **{ap.fr_nombre(zone['seuil_haut'], 2)}**, "
            "l'application affiche « incertain » au lieu de trancher."
        )
        x, y = st.columns(2)
        x.metric("Hors de la zone", ap.fr_pct(zone["exactitude_hors_zone"]))
        y.metric("Dans la zone", ap.fr_pct(zone["exactitude_dans_la_zone"]))
        st.markdown(
            f"La zone contient **{ap.fr_pct(zone['part_dans_la_zone'])}** des critiques du test, soit "
            f"{ap.fr_entier(zone['effectif_dans_la_zone'])}. Les seuils ont été fixés avant d'ouvrir le test."
        )

with ui.carte("detail"):
    st.subheader("Détail par modèle", anchor=False)
    st.table(pd.DataFrame({
        LIBELLES_FAMILLES[f]: {
            "Exactitude": ap.fr_pct(m["exactitude"], 2), "Précision": ap.fr_pct(m["precision"], 2),
            "Rappel": ap.fr_pct(m["rappel"], 2), "F1": ap.fr_nombre(m["f1"], 3), "ROC-AUC": ap.fr_nombre(m["roc_auc"], 3),
        } for f, m in modeles.items()
    }).T)

ui.pied(artefacts, "Résultats du test, lus dans reports/metrics.json.")

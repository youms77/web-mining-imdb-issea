"""Page « Performances des modèles » : exactitude, confusion, ROC, calibration et zone d'incertitude."""
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
agr = ui.agregats()
noms_roc = {**{f: LIBELLES_FAMILLES[f] for f in modeles}, "vader_compound": LIBELLES_FAMILLES.get("vader", "VADER")}

ui.entete("Évaluation comparative des modèles de classification du sentiment")

ui.indicateurs([
    ("Exactitude", ap.fr_pct(resume["exactitude"], 2), "vert",
     f"IC 95 % : {ap.fr_pct(resume['ic95']['exactitude']['bas'], 2)} à {ap.fr_pct(resume['ic95']['exactitude']['haut'], 2)}"),
    ("Score F1", ap.fr_nombre(resume["f1"], 3), "bleu"),
    ("ROC-AUC", ap.fr_nombre(resume["roc_auc"], 3), "orange"),
    ("Calibration (ECE)", ap.fr_nombre(resume["ece"], 4), "rouge"),
])

col_gauche, col_droite = st.columns([1.1, 1], gap="large")
with col_gauche:
    with ui.carte("comparaison"):
        st.subheader("Exactitude des modèles sur le jeu de test", anchor=False)
        lignes = [
            {"nom": LIBELLES_FAMILLES[f], "exactitude": modeles[f]["exactitude"], "bas": modeles[f]["ic95"]["exactitude"]["bas"],
             "haut": modeles[f]["ic95"]["exactitude"]["haut"], "retenu": f == retenu}
            for f in modeles
        ]
        ui.afficher(ui.points_modeles(lignes), "comparaison_modeles", "Exactitude de chaque modèle sur le test, avec son intervalle à 95 %.")
with col_droite:
    with ui.carte("confusion", ton="vert"):
        st.subheader("Matrice de confusion", anchor=False)
        famille = st.selectbox("Modèle", list(modeles), index=list(modeles).index(retenu), format_func=LIBELLES_FAMILLES.get, key="modele_confusion")
        ui.afficher(ui.matrice_confusion(modeles[famille]["matrice_confusion"]), "confusion", "Matrice de confusion du modèle choisi.")

col_roc, col_calib = st.columns([1.1, 1], gap="large")
with col_roc:
    with ui.carte("roc", ton="orange"):
        st.subheader("Courbes ROC des modèles sur le jeu de test", anchor=False)
        ui.afficher(ui.courbes_roc(agr["roc"], noms_roc, retenu), "roc", "Courbes ROC des cinq modèles superposées, le SVM linéaire retenu en bleu épais.")
with col_calib:
    with ui.carte("calibration"):
        st.subheader("Courbe de calibration du SVM linéaire", anchor=False)
        ui.afficher(ui.courbe_calibration(test["fiabilite_modele_retenu"]), "calibration", "Courbe de calibration du SVM sur le test.")

with ui.carte("zone", ton="orange"):
    st.subheader("Exactitude selon la probabilité estimée et zone d'incertitude", anchor=False)
    ui.indicateurs([
        ("Exactitude hors de la zone", ap.fr_pct(zone["exactitude_hors_zone"]), "vert"),
        ("Exactitude dans la zone", ap.fr_pct(zone["exactitude_dans_la_zone"]), "orange"),
        ("Part des critiques dans la zone", ap.fr_pct(zone["part_dans_la_zone"]), "bleu", f"{ap.fr_entier(zone['effectif_dans_la_zone'])} critiques"),
    ])
    ui.afficher(ui.zone_incertitude(agr["incertitude"]), "zone", "Exactitude du SVM par tranche de probabilité, zone d'incertitude en orange.")

with ui.carte("detail"):
    st.subheader("Métriques détaillées par modèle", anchor=False)
    st.table(pd.DataFrame({
        LIBELLES_FAMILLES[f]: {
            "Exactitude": ap.fr_pct(m["exactitude"], 2), "Précision": ap.fr_pct(m["precision"], 2),
            "Rappel": ap.fr_pct(m["rappel"], 2), "F1": ap.fr_nombre(m["f1"], 3), "ROC-AUC": ap.fr_nombre(m["roc_auc"], 3),
        } for f, m in modeles.items()
    }).T)

ui.commentaire(artefacts, [
    f"**Ce qui est présenté.** Évaluation unique de cinq modèles sur {ap.fr_entier(n_test)} critiques jamais vues pendant "
    f"l'entraînement : trois modèles appris (Naive Bayes, régression logistique, SVM linéaire), la référence VADER et le SVM "
    f"calibré retenu ({resume['modele'].replace('sigmoid', 'sigmoïde')}). Les seuils et choix ont été fixés avant l'ouverture du test.",
    "**Comment lire.** L'exactitude est la part de critiques bien classées, avec son intervalle de confiance à 95 % (l'axe ne part pas de zéro). "
    "La matrice de confusion donne, dans chaque case, l'effectif et la part de la vraie classe. Une courbe ROC proche du coin supérieur gauche "
    "indique un bon classement ; la diagonale correspond au hasard. La courbe de calibration compare la probabilité annoncée à la part réellement "
    f"positive. Entre {ap.fr_nombre(zone['seuil_bas'], 2)} et {ap.fr_nombre(zone['seuil_haut'], 2)}, l'application affiche « incertain » "
    "au lieu de trancher ; l'exactitude y est nettement plus basse.",
    "**Calcul.** Les courbes ROC et le graphique de la zone d'incertitude sont tracés à partir des prédictions de test déjà enregistrées : "
    "aucun modèle n'est réentraîné et aucun choix ne change.",
])

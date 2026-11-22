"""Page « Accueil » : présentation du projet, du pipeline et de l'équipe."""
from __future__ import annotations

import base64
from html import escape

import streamlit as st

import interface as ui
from src import application as ap

artefacts = ui.charger()
retenues = artefacts.metriques["donnees"]["corpus_retenu"]["lignes"]
brutes = artefacts.metriques["donnees"]["brut"]["lignes"]


@st.cache_data(show_spinner=False)
def _image_base64(chemin: str) -> str:
    with open(chemin, "rb") as f:
        return base64.b64encode(f.read()).decode()


fond = _image_base64(str(ui.DOSSIER_ASSETS / ui.FOND_ACCUEIL))
logo = _image_base64(str(ui.DOSSIER_ASSETS / "logo_ISSEA.png"))
st.html(
    '<div class="heros">'
    f'<div class="fond" style="background-image:url(data:image/jpeg;base64,{fond})"></div>'
    '<div class="contenu">'
    f'<img class="logo" alt="Logo de l\'ISSEA" src="data:image/png;base64,{logo}">'
    "<h1>Exploration des thèmes et prédiction de la satisfaction dans les critiques de films</h1>"
    '<div class="sous">Application des méthodes de Web Mining</div>'
    "</div></div>"
)

ETAPES = [
    ("🌐", "Collecte des données"), ("🧹", "Nettoyage"), ("✂️", "Normalisation et tokenisation"),
    ("🔢", "Vectorisation"), ("🧩", "Modélisation thématique"), ("😊", "Analyse des sentiments"), ("💻", "Application"),
]
morceaux = []
for i, (icone, libelle) in enumerate(ETAPES):
    if i:
        morceaux.append('<div class="fleche">→</div>')
    morceaux.append(f'<div class="etape"><div class="ico">{icone}</div><div class="lib">{escape(libelle)}</div></div>')
st.html(f'<div class="pipeline">{"".join(morceaux)}</div>')

st.html(
    '<div class="contexte">'
    "Cette application a été réalisée dans le cadre du cours de Web Mining du Dr TCHAGNA, à l'Institut Sous-régional de "
    "Statistique et d'Économie Appliquée (ISSEA) de Yaoundé, en novembre 2026. Elle exploite le corpus « IMDB Dataset of 50K "
    "Movie Reviews », diffusé sur Kaggle et issu des travaux de Maas et al. (2011) : "
    f"{ap.fr_entier(brutes)} critiques de films rédigées en anglais par des internautes sur le site IMDb, réparties à parts égales "
    f"entre avis positifs et négatifs. Après suppression des doublons, {ap.fr_entier(retenues)} critiques ont été analysées afin "
    "d'identifier les thèmes associés aux avis favorables ou défavorables et d'estimer le sentiment d'une nouvelle critique."
    "</div>"
)

MEMBRES = [
    ("YOUMBI BILLONG Joël", "Chef de groupe", "YJ", "#1E40AF"),
    ("ATEBA JOSEPH Stève", "", "AS", "#16A34A"),
    ("MBO NDONG Canuto Nzé", "", "MC", "#EA580C"),
    ("NGANONGO Juste Adji", "", "NJ", "#7C3AED"),
]
cartes = "".join(
    f'<div class="membre"><div class="avatar" style="background:{couleur}">{initiales}</div>'
    f'<div class="nom">{escape(nom)}</div>{f"<div class=role>{escape(role)}</div>" if role else ""}</div>'
    for nom, role, initiales, couleur in MEMBRES
)
st.html(f'<div class="equipe">{cartes}</div>')
st.html('<div class="mention">Élèves ingénieurs statisticiens économistes (ISE), classe ISE3 Data Science et Marketing</div>')

ui.commentaire(artefacts, [
    "**Ce qui est présenté.** La page d'accueil rappelle le titre du projet, le cadre du cours, la chaîne de traitement "
    "suivie, de la collecte des données à l'application, et l'équipe qui l'a réalisée.",
    "**Comment la lire.** Les sept étapes se lisent de gauche à droite. Les pages du menu à gauche détaillent le corpus, "
    "les thèmes, les performances des modèles, la prédiction d'une nouvelle critique et la méthodologie.",
    "**Photographie de fond.** « Salle cinéma 3049 », par HDewaere, "
    "[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Salle_cin%C3%A9ma_3049.jpg), "
    "licence [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Image recadrée.",
])

"""Point d'entrée de l'application Streamlit : navigation entre les six pages.

Lancement depuis la racine du projet :
    .\\.venv\\Scripts\\python.exe -m streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

DOSSIER_APP = Path(__file__).resolve().parent
for dossier in (DOSSIER_APP, DOSSIER_APP.parent):
    if str(dossier) not in sys.path:
        sys.path.insert(0, str(dossier))

import streamlit as st  # noqa: E402

from interface import appliquer_style, charger  # noqa: E402

LIEN_VIDEO_DEMO = "https://drive.google.com/file/d/14S__DxeDey1j1oeA-jtwNp9-PzIogFDn/view?usp=sharing"

st.set_page_config(page_title="Critiques IMDB", page_icon=":material/movie:", layout="wide")
appliquer_style()
st.logo(str(DOSSIER_APP / "assets" / "logo_ISSEA.png"), size="large")

# Chargement unique au démarrage : le premier clic sur « Analyser » n'attend pas les modèles.
charger()

pages = [
    st.Page("vues/accueil.py", title="Accueil", icon=":material/home:", default=True),
    st.Page("vues/descriptive.py", title="Analyse descriptive", icon=":material/bar_chart:"),
    st.Page("vues/explorer.py", title="Exploration des thèmes", icon=":material/category:"),
    st.Page("vues/performances.py", title="Performances des modèles", icon=":material/monitoring:"),
    st.Page("vues/analyser.py", title="Analyse d'une nouvelle critique", icon=":material/rate_review:"),
    st.Page("vues/methode.py", title="Méthodologie et limites", icon=":material/menu_book:"),
]
page = st.navigation(pages)
with st.sidebar:
    st.markdown("**Critiques IMDB**")
    st.markdown("Web Mining, ISSEA, ISE3 Data Science et Marketing")
    # Bouton de la vidéo de démonstration, placé en bas de la barre latérale (style dans interface.py).
    st.link_button("Vidéo de démonstration", LIEN_VIDEO_DEMO, icon=":material/play_circle:", use_container_width=True)
page.run()

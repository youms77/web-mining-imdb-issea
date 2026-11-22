"""Point d'entrée de l'application Streamlit : navigation entre les quatre pages.

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

st.set_page_config(page_title="Critiques IMDB", page_icon=":material/movie:", layout="wide")
appliquer_style()
st.logo(str(DOSSIER_APP / "assets" / "logo_ISSEA.png"), size="large")

# Chargement unique au démarrage : le premier clic sur « Analyser » n'attend pas les modèles.
charger()

pages = [
    st.Page("vues/analyser.py", title="Analyser une critique", icon=":material/rate_review:", default=True),
    st.Page("vues/explorer.py", title="Explorer les thèmes", icon=":material/category:"),
    st.Page("vues/performances.py", title="Performances", icon=":material/monitoring:"),
    st.Page("vues/methode.py", title="Méthode et limites", icon=":material/menu_book:"),
]
page = st.navigation(pages)
with st.sidebar:
    st.markdown("**Critiques IMDB**")
    st.caption("Thèmes et sentiment. Projet de Web Mining, ISSEA, ISE3-DSM.")
page.run()

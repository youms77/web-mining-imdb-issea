"""Éléments d'interface partagés : style, chargement, formats et graphiques Plotly."""
from __future__ import annotations

import sys
from html import escape
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import application as ap  # noqa: E402

BLEU, VERT, ROUGE, ORANGE = "#1F5FBF", "#1B7F4B", "#C8372D", "#E07B00"
ENCRE, SECONDAIRE, FILET, GRIS = "#17202E", "#4A5568", "#DDE2EA", "#A9B2C0"
TAILLE_MIN = 16
CONFIG_PLOTLY = {"displayModeBar": False, "responsive": True, "scrollZoom": False}

_CSS = """
<style>
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] p,
[data-testid="stSidebarNav"] span, [data-testid="stTable"] td, [data-testid="stTable"] th,
[data-testid="stBadge"] p, [data-baseweb="tab"] p, [data-testid="stExpander"] summary p,
button p, textarea, input, [data-baseweb="select"] div { font-size: 1rem !important; }
.block-container { padding-top: 2.4rem; max-width: 1240px; }
h1 { letter-spacing: -0.01em; }
[class*="st-key-carte_"], [data-testid="stMetric"] { background: #FFFFFF; border-radius: 10px; }
@media (max-width: 640px) { h1 { font-size: 2rem !important; } }
.chapeau { color: #4A5568; font-size: 1.16rem; line-height: 1.5; max-width: 54rem; margin: 0 0 .4rem; }
.texte-analyse { font-size: 1.06rem; line-height: 1.75; white-space: pre-wrap; overflow-wrap: anywhere; }
mark.mot-pos { background: #DDF3E6; color: #17202E; border-bottom: 2px solid #1B7F4B; padding: 0 .12em; border-radius: 3px; }
mark.mot-neg { background: #FBE3E0; color: #17202E; border-bottom: 2px dotted #C8372D; padding: 0 .12em; border-radius: 3px; }
.legende { display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; color: #4A5568; font-size: 1rem; margin-bottom: .5rem; }
.theme-ligne { margin: 0 0 1.05rem; }
.theme-tete { display: flex; justify-content: space-between; gap: .8rem; font-weight: 600; font-size: 1.04rem; }
.theme-barre { height: 8px; background: #EDF0F5; border-radius: 4px; margin: .3rem 0 .3rem; overflow: hidden; }
.theme-barre span { display: block; height: 100%; background: #1F5FBF; border-radius: 4px; }
.theme-mots { color: #4A5568; font-size: 1rem; }
.extrait { border-left: 3px solid #1F5FBF; padding: .15rem 0 .15rem .9rem; color: #2D3748; font-style: italic; font-size: 1.02rem; }
.pied { color: #4A5568; font-size: 1rem; border-top: 1px solid #DDE2EA; margin-top: 2rem; padding-top: .7rem; }
</style>
"""


def appliquer_style() -> None:
    """CSS court : ce que le thème natif ne permet pas de régler."""
    st.html(_CSS)


@st.cache_resource(show_spinner="Chargement des modèles (une seule fois)…")
def _charger_artefacts() -> ap.Artefacts:
    return ap.charger_artefacts()


def charger() -> ap.Artefacts:
    """Renvoie les modèles chargés, ou arrête la page avec un message clair."""
    try:
        return _charger_artefacts()
    except ap.ErreurChargement as erreur:
        st.error(
            f"Échec de chargement : {erreur} Vérifiez que `models/deploy/` et `reports/metrics.json` sont présents, "
            "puis relancez l'application.",
            icon=":material/error:",
        )
        st.stop()


# --- Textes -------------------------------------------------------------------------------------

def fr_proba(p: float) -> str:
    """Probabilité à trois décimales, sans afficher un faux 0 ou un faux 1."""
    if p >= 0.9995:
        return "> 0,999"
    if p < 0.0005:
        return "< 0,001"
    return ap.fr_nombre(p, 3)


def fr_ecart(valeur: float, decimales: int = 1) -> str:
    """Écart signé en points, avec un vrai signe moins."""
    signe = "+" if valeur > 0 else ("−" if valeur < 0 else "")
    return f"{signe}{ap.fr_nombre(abs(valeur), decimales)} pt"


def carte(nom: str, parent=None, **options):
    """Conteneur à bordure sur fond blanc. La clé stable sert de seul crochet CSS."""
    return (parent or st).container(border=True, key=f"carte_{nom}", **options)


def entete(titre: str, chapeau: str) -> None:
    st.title(titre, anchor=False)
    st.html(f'<p class="chapeau">{escape(chapeau)}</p>')


def pied(artefacts: ap.Artefacts, complement: str = "") -> None:
    """Source et date en bas de chaque vue."""
    date = artefacts.metriques["sentiment"]["date_gel_des_choix"][:10]
    jour = f"{date[8:10]}/{date[5:7]}/{date[:4]}"
    texte = f"Source : corpus IMDB de 50 000 critiques (Maas et al., 2011), via Kaggle. Modèle gelé le {jour}."
    st.html(f'<p class="pied">{escape(texte)}{" " + escape(complement) if complement else ""}</p>')


# --- Graphiques ---------------------------------------------------------------------------------

def _style(figure: go.Figure, hauteur: int, marges: dict | None = None) -> go.Figure:
    figure.update_layout(
        height=hauteur,
        margin=marges or {"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Source Sans, Source Sans Pro, sans-serif", "size": TAILLE_MIN, "color": ENCRE},
        hoverlabel={"font_size": TAILLE_MIN, "bgcolor": "#FFFFFF", "bordercolor": FILET, "font_color": ENCRE},
        separators=", ",
        showlegend=False,
        dragmode=False,
    )
    return figure


def afficher(figure: go.Figure, cle: str, description: str) -> None:
    st.plotly_chart(figure, key=cle, theme=None, config=CONFIG_PLOTLY, alt=description)


def jauge(p: float, bas: float, haut: float) -> go.Figure:
    """Barre 0-1 en trois zones, avec un repère sur le score."""
    figure = go.Figure()
    for debut, fin, couleur in ((0, bas, ROUGE), (bas, haut, ORANGE), (haut, 1, VERT)):
        figure.add_shape(type="rect", x0=debut + 0.003, x1=fin - 0.003, y0=0.2, y1=0.8, fillcolor=couleur, line_width=0, layer="below")
    figure.add_trace(go.Scatter(
        x=[p], y=[1.2], mode="markers", hoverinfo="skip",
        marker={"symbol": "triangle-down", "size": 20, "color": ENCRE, "line": {"color": "#FFFFFF", "width": 2}},
    ))
    figure.update_xaxes(
        range=[-0.01, 1.01], tickvals=[0, bas, haut, 1],
        ticktext=["0", ap.fr_nombre(bas, 2), ap.fr_nombre(haut, 2), "1"],
        showgrid=False, zeroline=False, fixedrange=True, ticks="",
    )
    figure.update_yaxes(range=[-0.1, 1.9], visible=False, fixedrange=True)
    return _style(figure, 92, {"l": 10, "r": 10, "t": 0, "b": 30})


def barres_divergentes(lignes: list[dict]) -> go.Figure:
    """Écart de présence entre avis positifs et négatifs, avec intervalle de confiance à 95 %."""
    lignes = sorted(lignes, key=lambda l: l["ecart_points"])
    y = list(range(len(lignes)))
    ecarts = [l["ecart_points"] for l in lignes]
    figure = go.Figure(go.Bar(
        x=ecarts, y=y, orientation="h", width=0.3,
        marker={"color": [VERT if e > 0 else ROUGE for e in ecarts], "cornerradius": 3},
        error_x={
            "type": "data", "symmetric": False, "color": ENCRE, "thickness": 1.3, "width": 3,
            "array": [l["ic95_haut"] - l["ecart_points"] for l in lignes],
            "arrayminus": [l["ecart_points"] - l["ic95_bas"] for l in lignes],
        },
        customdata=[
            [l["nom"], fr_ecart(l["ecart_points"]), ap.fr_nombre(l["ic95_bas"], 1), ap.fr_nombre(l["ic95_haut"], 1)]
            for l in lignes
        ],
        hovertemplate="<b>%{customdata[0]}</b><br>Écart : %{customdata[1]}<br>IC 95 % : %{customdata[2]} à %{customdata[3]}<extra></extra>",
    ))
    borne = max(max(abs(l["ic95_bas"]), abs(l["ic95_haut"])) for l in lignes) * 1.08
    for i, l in enumerate(lignes):
        figure.add_annotation(
            x=-borne, y=i + 0.27, xanchor="left", yanchor="bottom", showarrow=False, align="left",
            text=f"<b>{escape(l['nom'])}</b>  {fr_ecart(l['ecart_points'])}", font={"size": TAILLE_MIN, "color": ENCRE},
            bgcolor="#FFFFFF",
        )
    figure.add_vline(x=0, line_color=SECONDAIRE, line_width=1)
    figure.update_xaxes(
        range=[-borne, borne], title_text="← plus présent dans les avis négatifs · positifs →",
        title_font_size=TAILLE_MIN, gridcolor=FILET, zeroline=False, fixedrange=True, ticksuffix=" pt",
    )
    figure.update_yaxes(visible=False, range=[-0.5, len(lignes) - 0.05], fixedrange=True)
    return _style(figure, 66 * len(lignes) + 66, {"l": 8, "r": 8, "t": 4, "b": 58})


def points_modeles(lignes: list[dict]) -> go.Figure:
    """Exactitude sur le test avec intervalle à 95 %. Le modèle retenu ressort en bleu."""
    lignes = sorted(lignes, key=lambda l: l["exactitude"])
    y = list(range(len(lignes)))
    figure = go.Figure(go.Scatter(
        x=[l["exactitude"] for l in lignes], y=y, mode="markers",
        marker={"size": 14, "color": [BLEU if l["retenu"] else GRIS for l in lignes], "line": {"color": "#FFFFFF", "width": 2}},
        error_x={
            "type": "data", "symmetric": False, "color": SECONDAIRE, "thickness": 1.5, "width": 4,
            "array": [l["haut"] - l["exactitude"] for l in lignes],
            "arrayminus": [l["exactitude"] - l["bas"] for l in lignes],
        },
        customdata=[[l["nom"], ap.fr_pct(l["exactitude"], 2), ap.fr_pct(l["bas"], 2), ap.fr_pct(l["haut"], 2)] for l in lignes],
        hovertemplate="<b>%{customdata[0]}</b><br>Exactitude : %{customdata[1]}<br>IC 95 % : %{customdata[2]} à %{customdata[3]}<extra></extra>",
    ))
    bas = min(l["bas"] for l in lignes) - 0.02
    for i, l in enumerate(lignes):
        suffixe = " (retenu)" if l["retenu"] else ""
        figure.add_annotation(
            x=bas, y=i + 0.2, xanchor="left", yanchor="bottom", showarrow=False,
            text=f"<b>{escape(l['nom'])}</b>{suffixe}  {ap.fr_pct(l['exactitude'], 2)}", font={"size": TAILLE_MIN, "color": ENCRE},
            bgcolor="#FFFFFF",
        )
    graduations = [v / 100 for v in range(int(bas * 100) // 5 * 5 + 5, 95, 5)]
    figure.update_xaxes(range=[bas, 0.95], tickvals=graduations, ticktext=[f"{round(v * 100)}\u00a0%" for v in graduations], gridcolor=FILET, zeroline=False, fixedrange=True, title_text="Exactitude sur le test")
    figure.update_yaxes(visible=False, range=[-0.45, len(lignes) - 0.1], fixedrange=True)
    return _style(figure, 66 * len(lignes) + 60, {"l": 8, "r": 12, "t": 4, "b": 58})


def matrice_confusion(m: dict) -> go.Figure:
    """Matrice 2 × 2, en bleus ordonnés, avec effectif et part de la ligne."""
    valeurs = [[m["vrais_negatifs"], m["faux_positifs"]], [m["faux_negatifs"], m["vrais_positifs"]]]
    colonnes, lignes = ["Prédit négatif", "Prédit positif"], ["Vrai négatif", "Vrai positif"]
    texte = [
        [f"<b>{ap.fr_entier(v)}</b><br>{ap.fr_pct(v / sum(ligne), 1)}" for v in ligne]
        for ligne in valeurs
    ]
    maximum = max(max(l) for l in valeurs)
    figure = go.Figure(go.Heatmap(
        z=valeurs, x=colonnes, y=lignes, showscale=False, xgap=3, ygap=3, zmin=0, zmax=maximum,
        colorscale=[[0, "#EAF1FB"], [0.5, "#8FB4E8"], [1, "#1F5FBF"]],
        customdata=texte, hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>",
    ))
    for i, ligne in enumerate(valeurs):
        for j, v in enumerate(ligne):
            figure.add_annotation(
                x=colonnes[j], y=lignes[i], text=texte[i][j], showarrow=False,
                font={"size": 17, "color": "#FFFFFF" if v > 0.6 * maximum else ENCRE},
            )
    figure.update_xaxes(side="top", fixedrange=True, showgrid=False, ticks="")
    figure.update_yaxes(autorange="reversed", fixedrange=True, showgrid=False, automargin=True, ticks="")
    return _style(figure, 300, {"l": 8, "r": 8, "t": 36, "b": 8})


def courbe_calibration(fiabilite: list[dict]) -> go.Figure:
    """Probabilité annoncée contre fréquence observée, avec la diagonale de référence."""
    utiles = [t for t in fiabilite if t["effectif"] > 0]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Calibration parfaite",
        line={"color": GRIS, "width": 2}, hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        x=[t["probabilite_moyenne"] for t in utiles], y=[t["frequence_observee"] for t in utiles],
        mode="lines+markers", name="SVM calibré", line={"color": BLEU, "width": 2},
        marker={"size": 10, "color": BLEU, "line": {"color": "#FFFFFF", "width": 2}},
        customdata=[
            [ap.fr_nombre(t["probabilite_moyenne"], 3), ap.fr_pct(t["frequence_observee"]), ap.fr_entier(t["effectif"])]
            for t in utiles
        ],
        hovertemplate="Annoncé : %{customdata[0]}<br>Observé : %{customdata[1]}<br>%{customdata[2]} critiques<extra></extra>",
    ))
    figure.update_xaxes(range=[0, 1], dtick=0.2, tickformat=".1f", gridcolor=FILET, zeroline=False, fixedrange=True, title_text="Probabilité annoncée")
    figure.update_yaxes(range=[0, 1], tickvals=[0.2, 0.4, 0.6, 0.8, 1], tickformat=".1f", gridcolor=FILET, zeroline=False, fixedrange=True, title_text="Part réellement positive")
    _style(figure, 400, {"l": 74, "r": 14, "t": 44, "b": 70})
    figure.update_layout(showlegend=True, legend={"orientation": "h", "x": 0, "y": 1.13, "font_size": TAILLE_MIN})
    return figure

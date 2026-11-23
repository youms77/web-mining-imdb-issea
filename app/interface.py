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

BLEU, VERT, ROUGE, ORANGE = "#2563EB", "#16A34A", "#DC2626", "#EA580C"
BLEU_FONCE, BLEU_CLAIR = "#1E40AF", "#60A5FA"
ENCRE, SECONDAIRE, FILET, GRIS = "#0F172A", "#334155", "#DBEAFE", "#94A3B8"
TAILLE_MIN = 16
CONFIG_PLOTLY = {"displayModeBar": False, "responsive": True, "scrollZoom": False}
# Image de fond de l'accueil : un seul paramètre à changer (1 à 4, voir app/assets/fonds_candidats/apercu.png).
FOND_ACCUEIL = "fond_accueil.jpg"
DOSSIER_ASSETS = Path(__file__).resolve().parent / "assets"
TONS = {  # fond clair, couleur franche
    "bleu": ("#DBEAFE", "#2563EB"), "vert": ("#DCFCE7", "#16A34A"),
    "rouge": ("#FEE2E2", "#DC2626"), "orange": ("#FFEDD5", "#EA580C"),
}

_CSS = """
<style>
.stApp { background: linear-gradient(180deg, #EFF6FF 0%, #F8FAFC 100%); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] p,
[data-testid="stSidebarNav"] span, [data-testid="stTable"] td, [data-testid="stTable"] th,
[data-testid="stBadge"] p, [data-baseweb="tab"] p, [data-testid="stExpander"] summary p,
button p, textarea, input, [data-baseweb="select"] div { font-size: 1rem !important; }
.block-container { padding-top: 4.4rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: #2563EB; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebarHeader"] img { background: #FFFFFF; padding: 6px 10px; border-radius: 10px; }
[data-testid="stSidebarNav"] a { border-radius: 10px; margin-bottom: 4px; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: #1E40AF; font-weight: 700; }
[data-testid="stSidebarNav"] a:hover { background: rgba(255,255,255,.18); }
[data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] { margin-top: auto; }
[data-testid="stSidebar"] [data-testid^="stBaseLinkButton"] { background: #1E40AF; border: 1px solid #FFFFFF; border-radius: 10px; font-weight: 700; }
[data-testid="stSidebar"] [data-testid^="stBaseLinkButton"]:hover { background: #FFFFFF; }
[data-testid="stSidebar"] [data-testid^="stBaseLinkButton"]:hover * { color: #1E40AF !important; }
.bandeau { background: #1E40AF; color: #FFFFFF; border-radius: 14px; padding: 1.25rem 1.8rem; margin: 0 0 1.3rem;
  box-shadow: 0 4px 14px rgba(30,64,175,.28); }
.bandeau h1 { color: #FFFFFF !important; font-size: 2rem; line-height: 1.25; margin: 0; padding: 0; letter-spacing: -0.01em; }
[class*="st-key-carte_"] { background: #FFFFFF; border: 1px solid #BFDBFE !important; border-top: 5px solid #2563EB !important;
  border-radius: 12px; box-shadow: 0 2px 10px rgba(37,99,235,.10); }
[class*="st-key-carte_vert_"] { border-top-color: #16A34A !important; }
[class*="st-key-carte_rouge_"] { border-top-color: #DC2626 !important; }
[class*="st-key-carte_orange_"] { border-top-color: #EA580C !important; }
[data-testid="stMetric"] { background: #FFFFFF; border-radius: 10px; }
h2, h3 { color: #1E3A8A !important; }
@media (max-width: 640px) { .bandeau h1 { font-size: 1.5rem !important; } }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1rem; margin: 0 0 1.2rem; }
.kpi { border-radius: 12px; padding: .9rem 1.1rem; border-left: 8px solid; }
.kpi .k-lib { font-size: 1rem; color: #334155; font-weight: 600; }
.kpi .k-val { font-size: 2.1rem; font-weight: 700; color: #0F172A; line-height: 1.2; }
.kpi .k-det { font-size: 1rem; color: #334155; }
.texte-analyse { font-size: 1.06rem; line-height: 1.75; white-space: pre-wrap; overflow-wrap: anywhere; }
mark.mot-pos { background: #BBF7D0; color: #0F172A; border-bottom: 2px solid #16A34A; padding: 0 .12em; border-radius: 3px; }
mark.mot-neg { background: #FECACA; color: #0F172A; border-bottom: 2px dotted #DC2626; padding: 0 .12em; border-radius: 3px; }
.legende { display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; color: #334155; font-size: 1rem; margin-bottom: .5rem; }
.theme-ligne { margin: 0 0 1.05rem; }
.theme-tete { display: flex; justify-content: space-between; gap: .8rem; font-weight: 600; font-size: 1.04rem; }
.theme-barre { height: 10px; background: #DBEAFE; border-radius: 5px; margin: .3rem 0 .3rem; overflow: hidden; }
.theme-barre span { display: block; height: 100%; background: #2563EB; border-radius: 5px; }
.theme-mots { color: #334155; font-size: 1rem; }
.extrait { border-left: 4px solid #2563EB; padding: .15rem 0 .15rem .9rem; color: #1E293B; font-style: italic; font-size: 1.02rem; }
[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid #BFDBFE; border-radius: 12px; margin-top: 1.5rem; }
[data-testid="stTable"] th { background: #DBEAFE; color: #1E3A8A; }
.heros { position: relative; border-radius: 16px; overflow: hidden; background: #1E40AF; color: #FFFFFF; padding: 2.4rem 2rem;
  text-align: center; box-shadow: 0 6px 20px rgba(30,64,175,.30); margin-bottom: 1.4rem; }
.heros .fond { position: absolute; inset: 0; background-size: cover; background-position: center; opacity: .3; }
.heros .contenu { position: relative; }
.heros img.logo { height: 84px; background: #FFFFFF; padding: 8px 14px; border-radius: 12px; margin-bottom: 1rem; }
.heros h1 { color: #FFFFFF !important; font-size: 2.15rem; line-height: 1.25; margin: 0 auto .6rem; max-width: 56rem; }
.heros .sous { font-size: 1.25rem; color: #DBEAFE; font-weight: 600; }
.pipeline { display: flex; flex-wrap: wrap; justify-content: center; align-items: stretch; gap: .5rem; margin: .4rem 0 1.4rem; }
.etape { background: #FFFFFF; border: 1px solid #BFDBFE; border-top: 5px solid #2563EB; border-radius: 12px; width: 138px;
  padding: .7rem .4rem; text-align: center; box-shadow: 0 2px 8px rgba(37,99,235,.10); }
.etape .ico { font-size: 2rem; line-height: 1.2; }
.etape .lib { font-size: 1rem; font-weight: 600; color: #1E3A8A; line-height: 1.25; }
.fleche { align-self: center; color: #2563EB; font-size: 1.6rem; font-weight: 700; }
.contexte { background: #FFFFFF; border-left: 8px solid #2563EB; border-radius: 12px; padding: 1.1rem 1.4rem; font-size: 1.1rem;
  line-height: 1.65; color: #0F172A; box-shadow: 0 2px 10px rgba(37,99,235,.10); margin-bottom: 1.4rem; }
.equipe { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1rem; margin-bottom: .8rem; }
.membre { background: #FFFFFF; border: 1px solid #BFDBFE; border-radius: 14px; padding: 1rem .6rem; text-align: center;
  box-shadow: 0 2px 10px rgba(37,99,235,.10); }
.avatar { width: 84px; height: 84px; border-radius: 50%; margin: 0 auto .6rem; display: flex; align-items: center; justify-content: center;
  color: #FFFFFF; font-size: 1.9rem; font-weight: 700; }
.membre .nom { font-weight: 700; font-size: 1.05rem; color: #0F172A; }
.membre .role { color: #1E40AF; font-weight: 600; font-size: 1rem; }
.mention { text-align: center; color: #1E3A8A; font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; }
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


@st.cache_data(show_spinner=False)
def agregats() -> dict:
    """Agrégats légers de la refonte (histogrammes, ROC, carte), préparés par src/refonte_artefacts.py."""
    import json

    return json.loads((RACINE / "models" / "deploy" / "agregats_refonte.json").read_text(encoding="utf-8"))


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
    return f"{signe}{ap.fr_nombre(abs(valeur), decimales)} pt"


def carte(nom: str, parent=None, ton: str = "bleu", **options):
    """Conteneur à bordure sur fond blanc, bandeau coloré en haut. La clé sert de crochet CSS."""
    return (parent or st).container(border=True, key=f"carte_{ton}_{nom}", **options)


def entete(titre: str) -> None:
    """Bandeau de titre bleu foncé, texte blanc, en haut de chaque page."""
    st.html(f'<div class="bandeau"><h1>{escape(titre)}</h1></div>')


def indicateurs(lignes: list[tuple]) -> None:
    """Rangée de cartes colorées : (libellé, valeur, ton) ou (libellé, valeur, ton, détail)."""
    cartes = ""
    for ligne in lignes:
        libelle, valeur, ton = ligne[:3]
        detail = f'<div class="k-det">{escape(ligne[3])}</div>' if len(ligne) > 3 else ""
        fond, bord = TONS[ton]
        cartes += (
            f'<div class="kpi" style="background:{fond};border-color:{bord}"><div class="k-lib">{escape(libelle)}</div>'
            f'<div class="k-val">{escape(str(valeur))}</div>{detail}</div>'
        )
    st.html(f'<div class="kpis">{cartes}</div>')


def date_gel(artefacts: ap.Artefacts) -> str:
    date = artefacts.metriques["sentiment"]["date_gel_des_choix"][:10]
    return f"{date[8:10]}/{date[5:7]}/{date[:4]}"


def commentaire(artefacts: ap.Artefacts, paragraphes: list[str]) -> None:
    """Unique bloc de commentaire de la page, fermé par défaut : contenu, lecture, source et date."""
    with st.expander("Commentaire de la page"):
        for texte in paragraphes:
            st.markdown(texte)
        st.markdown(
            "**Source et date.** Corpus « IMDB Dataset of 50K Movie Reviews » (Maas et al., 2011), diffusé sur Kaggle. "
            f"Choix de modélisation gelés le {date_gel(artefacts)}. Chiffres lus dans `reports/metrics.json`."
        )


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


def barres_divergentes(lignes: list[dict], hauteur_ligne: int = 66) -> go.Figure:
    """Écart de présence entre avis positifs et négatifs, avec intervalle de confiance à 95 %."""
    lignes = sorted(lignes, key=lambda l: l["ecart_points"])
    y = list(range(len(lignes)))
    ecarts = [l["ecart_points"] for l in lignes]
    figure = go.Figure(go.Bar(
        x=ecarts, y=y, orientation="h", width=0.3 if hauteur_ligne > 50 else 0.45,
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
    decalage = 0.27 if hauteur_ligne > 50 else 0.3
    for i, l in enumerate(lignes):
        figure.add_annotation(
            x=-borne, y=i + decalage, xanchor="left", yanchor="bottom", showarrow=False, align="left",
            text=f"<b>{escape(l['nom'])}</b>  {fr_ecart(l['ecart_points'])}", font={"size": TAILLE_MIN, "color": ENCRE},
            bgcolor="#FFFFFF",
        )
    figure.add_vline(x=0, line_color=SECONDAIRE, line_width=1)
    figure.update_xaxes(
        range=[-borne, borne], title_text="← plus présent dans les avis négatifs · positifs →",
        title_font_size=TAILLE_MIN, gridcolor=FILET, zeroline=False, fixedrange=True, ticksuffix=" pt",
    )
    figure.update_yaxes(visible=False, range=[-0.5, len(lignes) - 0.05], fixedrange=True)
    return _style(figure, hauteur_ligne * len(lignes) + 66, {"l": 8, "r": 8, "t": 4, "b": 58})


def points_modeles(lignes: list[dict]) -> go.Figure:
    """Exactitude sur le test avec intervalle à 95 %. Le modèle retenu ressort en bleu foncé."""
    lignes = sorted(lignes, key=lambda l: l["exactitude"])
    y = list(range(len(lignes)))
    figure = go.Figure(go.Scatter(
        x=[l["exactitude"] for l in lignes], y=y, mode="markers",
        marker={"size": 16, "color": [BLEU_FONCE if l["retenu"] else BLEU_CLAIR for l in lignes], "line": {"color": "#FFFFFF", "width": 2}},
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
    figure.update_xaxes(range=[bas, 0.95], tickvals=graduations, ticktext=[f"{round(v * 100)} %" for v in graduations], gridcolor=FILET, zeroline=False, fixedrange=True, title_text="Exactitude sur le test")
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
        colorscale=[[0, "#DBEAFE"], [0.5, "#60A5FA"], [1, "#1E40AF"]],
        customdata=texte, hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>",
    ))
    for i, ligne in enumerate(valeurs):
        for j, v in enumerate(ligne):
            figure.add_annotation(
                x=colonnes[j], y=lignes[i], text=texte[i][j], showarrow=False,
                font={"size": 17, "color": "#FFFFFF" if v > 0.5 * maximum else ENCRE},
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
        line={"color": GRIS, "width": 2, "dash": "dash"}, hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        x=[t["probabilite_moyenne"] for t in utiles], y=[t["frequence_observee"] for t in utiles],
        mode="lines+markers", name="SVM calibré", line={"color": BLEU, "width": 3},
        marker={"size": 11, "color": BLEU, "line": {"color": "#FFFFFF", "width": 2}},
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


# --- Graphiques ajoutés par la refonte ------------------------------------------------------------

def barres_classes(effectifs: dict) -> go.Figure:
    """Nombre de critiques par classe (vert : positif, rouge : négatif)."""
    noms, valeurs = ["Négatif", "Positif"], [effectifs["negative"], effectifs["positive"]]
    figure = go.Figure(go.Bar(
        x=noms, y=valeurs, marker={"color": [ROUGE, VERT], "cornerradius": 6}, width=0.55,
        text=[ap.fr_entier(v) for v in valeurs], textposition="inside", insidetextanchor="middle",
        textfont={"size": 18, "color": "#FFFFFF"},
        hovertemplate="%{x} : %{y:,.0f} critiques<extra></extra>",
    ))
    figure.update_yaxes(gridcolor=FILET, fixedrange=True, title_text="Nombre de critiques", tickformat=",.0f")
    figure.update_xaxes(fixedrange=True)
    return _style(figure, 360, {"l": 70, "r": 10, "t": 10, "b": 40})


def histogramme_longueurs(h: dict) -> go.Figure:
    """Nombre de critiques par tranche de 25 mots, une série par classe."""
    pas = h["pas_mots"]
    x = [b + pas / 2 for b in h["bornes"]]
    figure = go.Figure()
    for classe, nom, couleur in (("negative", "Négatif", ROUGE), ("positive", "Positif", VERT)):
        figure.add_trace(go.Bar(
            x=x, y=h["classes"][classe]["effectifs"], name=nom, width=pas, marker={"color": couleur}, opacity=0.62,
            hovertemplate=f"{nom} · %{{x:.0f}} mots (milieu de tranche)<br>%{{y:,.0f}} critiques<extra></extra>",
        ))
    figure.update_layout(barmode="overlay", bargap=0.02, showlegend=True, legend={"orientation": "h", "x": 0.55, "y": 0.98, "font_size": TAILLE_MIN})
    figure.update_xaxes(title_text="Nombre de mots par critique", fixedrange=True, gridcolor=FILET, range=[0, h["maximum_affiche"]])
    figure.update_yaxes(title_text="Nombre de critiques", fixedrange=True, gridcolor=FILET, tickformat=",.0f")
    _style(figure, 360, {"l": 70, "r": 10, "t": 10, "b": 56})
    figure.update_layout(showlegend=True)
    return figure


def barres_mots(lignes: list[dict], couleur: str) -> go.Figure:
    """Termes classés par valeur absolue du log-rapport de cotes."""
    lignes = sorted(lignes, key=lambda l: abs(l["delta"]))
    figure = go.Figure(go.Bar(
        x=[abs(l["delta"]) for l in lignes], y=[l["terme"] for l in lignes], orientation="h",
        marker={"color": couleur, "cornerradius": 3},
        customdata=[[ap.fr_entier(l["occurrences_positif"]), ap.fr_entier(l["occurrences_negatif"])] for l in lignes],
        hovertemplate="<b>%{y}</b><br>|log-rapport de cotes| : %{x:.2f}<br>Occurrences positives : %{customdata[0]}<br>Occurrences négatives : %{customdata[1]}<extra></extra>",
    ))
    figure.update_xaxes(title_text="|log-rapport de cotes|", gridcolor=FILET, fixedrange=True)
    figure.update_yaxes(fixedrange=True, automargin=True)
    return _style(figure, 30 * len(lignes) + 70, {"l": 8, "r": 10, "t": 6, "b": 52})


def barres_parts(lignes: list[tuple[str, float]]) -> go.Figure:
    """Part des critiques concernées, en pourcentage (barres horizontales bleues)."""
    lignes = sorted(lignes, key=lambda l: l[1])
    figure = go.Figure(go.Bar(
        x=[v for _, v in lignes], y=[n for n, _ in lignes], orientation="h", marker={"color": BLEU, "cornerradius": 3},
        text=[f"{ap.fr_nombre(v, 2)} %" for _, v in lignes], textposition="outside", cliponaxis=False,
        hovertemplate="%{y} : %{x:.2f} % des critiques<extra></extra>",
    ))
    figure.update_xaxes(title_text="Part des critiques (%)", gridcolor=FILET, fixedrange=True, range=[0, 75])
    figure.update_yaxes(fixedrange=True, automargin=True)
    return _style(figure, 60 * len(lignes) + 60, {"l": 8, "r": 70, "t": 6, "b": 52})


def courbes_roc(roc: dict, noms: dict, retenu: str) -> go.Figure:
    """ROC des modèles sur le test, le modèle retenu en bleu épais, la diagonale en tirets."""
    couleurs = {"naive_bayes_bow": ORANGE, "naive_bayes_tfidf": "#7C3AED", "regression_logistique": VERT, "vader_compound": GRIS}
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Hasard", line={"color": GRIS, "width": 1.5, "dash": "dash"}, hoverinfo="skip"))
    ordre = [m for m in roc["modeles"] if m != retenu] + [retenu]
    for m in ordre:
        d = roc["modeles"][m]
        est_retenu = m == retenu
        figure.add_trace(go.Scatter(
            x=d["fpr"], y=d["tpr"], mode="lines", name=f"{noms[m]} (AUC {ap.fr_nombre(d['auc'], 3)})",
            line={"color": BLEU_FONCE if est_retenu else couleurs[m], "width": 5 if est_retenu else 2.2},
            hovertemplate=f"<b>{noms[m]}</b><br>Faux positifs : %{{x:.3f}}<br>Vrais positifs : %{{y:.3f}}<extra></extra>",
        ))
    figure.update_xaxes(range=[0, 1], title_text="Taux de faux positifs", gridcolor=FILET, fixedrange=True, zeroline=False)
    figure.update_yaxes(range=[0, 1.02], title_text="Taux de vrais positifs", gridcolor=FILET, fixedrange=True, zeroline=False)
    _style(figure, 460, {"l": 70, "r": 10, "t": 10, "b": 60})
    figure.update_layout(showlegend=True, legend={"x": 0.42, "y": 0.04, "bgcolor": "rgba(255,255,255,.85)", "font_size": TAILLE_MIN})
    return figure


def zone_incertitude(inc: dict) -> go.Figure:
    """Exactitude du SVM par tranche de probabilité, avec la zone d'incertitude en orange."""
    tranches = [t for t in inc["tranches"] if t["effectif"] > 0]
    bas, haut = inc["seuil_bas"], inc["seuil_haut"]
    milieux = [(t["bas"] + t["haut"]) / 2 for t in tranches]
    couleurs = [ROUGE if m < bas else (VERT if m > haut else ORANGE) for m in milieux]
    figure = go.Figure()
    figure.add_vrect(x0=bas, x1=haut, fillcolor=ORANGE, opacity=0.16, line_width=0, layer="below")
    figure.add_trace(go.Bar(
        x=milieux, y=[t["corrects"] / t["effectif"] for t in tranches], width=0.045, marker={"color": couleurs, "cornerradius": 3},
        customdata=[[ap.fr_nombre(t["bas"], 2), ap.fr_nombre(t["haut"], 2), ap.fr_entier(t["effectif"]), ap.fr_pct(t["corrects"] / t["effectif"])] for t in tranches],
        hovertemplate="Probabilité de %{customdata[0]} à %{customdata[1]}<br>%{customdata[2]} critiques<br>Exactitude : %{customdata[3]}<extra></extra>",
    ))
    figure.add_annotation(
        x=(bas + haut) / 2, y=1.0, yref="paper", yanchor="bottom", showarrow=False, text="<b>Zone d'incertitude</b>",
        font={"size": TAILLE_MIN, "color": "#9A3412"},
    )
    figure.update_xaxes(range=[0, 1], dtick=0.1, tickformat=".1f", title_text="Probabilité estimée d'un avis positif", gridcolor=FILET, fixedrange=True, zeroline=False)
    figure.update_yaxes(range=[0, 1.0], tickformat=".0%", title_text="Exactitude de la prédiction", gridcolor=FILET, fixedrange=True, zeroline=False)
    return _style(figure, 400, {"l": 80, "r": 10, "t": 30, "b": 66})


def nuage_themes(points: list[dict], noms: dict) -> go.Figure:
    """Carte 2D des critiques (échantillon), colorées selon la polarité."""
    figure = go.Figure()
    for s, nom, couleur in ((0, "Avis négatifs", ROUGE), (1, "Avis positifs", VERT)):
        sel = [p for p in points if p["s"] == s]
        figure.add_trace(go.Scatter(
            x=[p["x"] for p in sel], y=[p["y"] for p in sel], mode="markers", name=nom,
            marker={"size": 6, "color": couleur, "opacity": 0.55},
            customdata=[noms[p["t"]] for p in sel],
            hovertemplate=f"{nom}<br>Thème : %{{customdata}}<extra></extra>",
        ))
    figure.update_xaxes(visible=False, fixedrange=True)
    figure.update_yaxes(visible=False, fixedrange=True)
    _style(figure, 470, {"l": 4, "r": 4, "t": 4, "b": 4})
    figure.update_layout(showlegend=True, legend={"orientation": "h", "x": 0, "y": 1.04, "font_size": TAILLE_MIN})
    return figure

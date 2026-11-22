"""Mesures et figures de l'exploration du corpus IMDB (phase 1).

Exemple d'utilisation :
    from src.data import charger_brut, dedoublonner
    from src.exploration import mesurer_notes
    corpus = dedoublonner(charger_brut())
    notes = mesurer_notes(corpus)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from src.data import CLASSES, RACINE, cle_normalisee

CHEMIN_METRICS = RACINE / "reports" / "metrics.json"
DOSSIER_FIGURES = RACINE / "reports" / "figures"

COULEURS = {"negative": "#e34948", "positive": "#2a78d6"}
LIBELLES = {"negative": "Négatif", "positive": "Positif"}
ENCRE, ENCRE_2, GRILLE, AXE, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"

# Seuils de l'article de Maas et al. (2011) : négatif si note <= 4, positif si note >= 7.
SEUIL_NEGATIF, SEUIL_POSITIF = 4.0, 7.0

MOTIFS_BRUIT = {
    "balise_br": r"<br\s*/?>",
    "autre_balise": r"</?(?!br\b)[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>",
    "entite_html": r"&(?:[a-zA-Z]+|#\d+);",
    "url": r"https?://|www\.",
}
_NOTE = re.compile(
    r"(?<![\d/])(?<!\d[.,])(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:/|\s+out\s+of\s+)\s*10(?![\d/]|[a-z])",
    flags=re.IGNORECASE,
)


def extraire_notes(texte: str) -> list[float]:
    """Renvoie les notes sur 10 écrites en chiffres, par exemple « 7/10 » ou « 3 out of 10 »."""
    return [float(m.group(1).replace(",", ".")) for m in _NOTE.finditer(texte)]


def mesurer_longueurs(corpus: pd.DataFrame) -> dict:
    """Résume la longueur en caractères (texte brut) et en mots (texte sans <br>), par classe."""
    mots = corpus["review"].map(cle_normalisee).str.split().str.len()
    caracteres = corpus["review"].str.len()
    resultat: dict = {}
    for classe in CLASSES:
        masque = corpus["sentiment"] == classe
        resultat[classe] = {
            "mots": _resumer(mots[masque]),
            "caracteres": _resumer(caracteres[masque]),
        }
    return resultat


def _resumer(serie: pd.Series) -> dict:
    return {
        "minimum": int(serie.min()),
        "q1": float(serie.quantile(0.25)),
        "mediane": float(serie.median()),
        "moyenne": round(float(serie.mean()), 1),
        "q3": float(serie.quantile(0.75)),
        "p95": float(serie.quantile(0.95)),
        "maximum": int(serie.max()),
    }


def mesurer_bruit(corpus: pd.DataFrame) -> dict:
    """Compte les critiques contenant chaque motif de bruit, au total et par classe."""
    resultat: dict = {}
    for nom, motif in MOTIFS_BRUIT.items():
        present = corpus["review"].str.contains(motif, regex=True)
        resultat[nom] = {
            "critiques": int(present.sum()),
            "part_pct": round(100 * float(present.mean()), 2),
            "par_classe": {c: int(present[corpus["sentiment"] == c].sum()) for c in CLASSES},
        }
    resultat["balise_br"]["occurrences"] = int(
        corpus["review"].str.count(MOTIFS_BRUIT["balise_br"]).sum()
    )
    autres = corpus["review"].str.findall(MOTIFS_BRUIT["autre_balise"]).explode().dropna()
    resultat["autre_balise"]["detail"] = {
        k: int(v) for k, v in autres.str.lower().value_counts().head(10).items()
    }
    return resultat


def mesurer_notes(corpus: pd.DataFrame) -> dict:
    """Mesure les notes écrites dans le texte et leur accord avec l'étiquette."""
    notes = corpus["review"].map(extraire_notes)
    avec_note = notes.str.len() > 0
    sous_corpus = pd.DataFrame({"notes": notes[avec_note], "sentiment": corpus.loc[avec_note, "sentiment"]})

    def sens(valeurs: list[float]) -> str:
        if any(v > 10 for v in valeurs):
            return "hors_echelle"
        if all(v <= SEUIL_NEGATIF for v in valeurs):
            return "negative"
        if all(v >= SEUIL_POSITIF for v in valeurs):
            return "positive"
        return "intermediaire_ou_mixte"

    sous_corpus["sens"] = sous_corpus["notes"].map(sens)
    nettes = sous_corpus[sous_corpus["sens"].isin(CLASSES)]
    accord = nettes["sens"] == nettes["sentiment"]

    par_valeur = {c: {str(v): 0 for v in range(11)} for c in CLASSES}
    for valeurs, classe in zip(sous_corpus["notes"], sous_corpus["sentiment"]):
        for entier in {int(v) for v in valeurs if v <= 10}:
            par_valeur[classe][str(entier)] += 1

    return {
        "critiques_avec_note": int(avec_note.sum()),
        "part_pct": round(100 * float(avec_note.mean()), 2),
        "par_classe": {c: int((sous_corpus["sentiment"] == c).sum()) for c in CLASSES},
        "repartition_sens": {k: int(v) for k, v in sous_corpus["sens"].value_counts().items()},
        "critiques_note_nette": int(len(nettes)),
        "accord_note_etiquette": int(accord.sum()),
        "accord_pct": round(100 * float(accord.mean()), 2),
        "critiques_par_note_entiere": par_valeur,
    }


def compter_mots(corpus: pd.DataFrame) -> pd.DataFrame:
    """Compte chaque mot par classe, après retrait des <br> et passage en minuscules."""
    vectoriseur = CountVectorizer(
        preprocessor=cle_normalisee,
        token_pattern=r"[^\W\d_]+(?:'[^\W\d_]+)?",
    )
    matrice = vectoriseur.fit_transform(corpus["review"])
    table = pd.DataFrame(index=vectoriseur.get_feature_names_out())
    for classe in CLASSES:
        masque = (corpus["sentiment"] == classe).to_numpy()
        table[classe] = np.asarray(matrice[masque].sum(axis=0)).ravel()
    return table


def mots_frequents(table: pd.DataFrame, n: int = 15) -> dict:
    """Mots les plus fréquents par classe (pour 1 000 mots), hors mots vides et contractions."""
    garder = ~table.index.isin(list(ENGLISH_STOP_WORDS)) & ~table.index.str.contains("'")
    resultat = {}
    for classe in CLASSES:
        taux = 1000 * table.loc[garder, classe] / table[classe].sum()
        resultat[classe] = {k: round(float(v), 2) for k, v in taux.nlargest(n).items()}
    return resultat


def mots_associes(table: pd.DataFrame, n: int = 12, min_occurrences: int = 500) -> dict:
    """Mots les plus liés à chaque classe : log2 du rapport des fréquences lissées."""
    freq = table[table.sum(axis=1) >= min_occurrences]
    freq = freq[~freq.index.str.contains("'")]
    ratio = np.log2(
        ((freq["positive"] + 1) / (table["positive"].sum() + len(table)))
        / ((freq["negative"] + 1) / (table["negative"].sum() + len(table)))
    )
    return {
        "min_occurrences": min_occurrences,
        "positive": {k: round(float(v), 3) for k, v in ratio.nlargest(n).items()},
        "negative": {k: round(float(v), 3) for k, v in ratio.nsmallest(n).items()},
    }


def enregistrer_metrics(cle: str, valeurs: dict, chemin: Path = CHEMIN_METRICS) -> None:
    """Écrit une section de metrics.json sans toucher aux autres sections."""
    contenu = json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else {}
    contenu[cle] = valeurs
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_suffix(".json.tmp")
    temporaire.write_text(json.dumps(contenu, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporaire, chemin)


# --- Figures : texte de 16 pt minimum, mêmes couleurs de classe partout ---

def fr(valeur: float, decimales: int = 0) -> str:
    """Formate un nombre à la française."""
    texte = f"{valeur:,.{decimales}f}"
    return texte.replace(",", " ").replace(".", ",")


def appliquer_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 16, "axes.titlesize": 21,
        "axes.labelsize": 17, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 16, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": ENCRE, "axes.labelcolor": ENCRE_2,
        "xtick.color": ENCRE_2, "ytick.color": ENCRE_2, "axes.edgecolor": AXE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": GRILLE, "grid.linewidth": 0.8, "axes.axisbelow": True,
    })


def _sauver(fig: plt.Figure, nom: str) -> plt.Figure:
    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOSSIER_FIGURES / nom, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def figure_longueurs(corpus: pd.DataFrame, longueurs: dict) -> plt.Figure:
    appliquer_style()
    mots = corpus["review"].map(cle_normalisee).str.split().str.len()
    bornes = np.geomspace(max(mots.min(), 1), mots.max() + 1, 46)
    fig, ax = plt.subplots(figsize=(13, 7))
    for classe in CLASSES:
        valeurs = mots[corpus["sentiment"] == classe]
        poids = np.full(len(valeurs), 100 / len(valeurs))
        mediane = longueurs[classe]["mots"]["mediane"]
        ax.hist(valeurs, bins=bornes, weights=poids, histtype="step", linewidth=2.4,
                color=COULEURS[classe], label=f"{LIBELLES[classe]} (médiane : {fr(mediane)} mots)")
    ax.set_xscale("log")
    ax.set_xlim(10, 2600)
    ax.set_xticks([10, 30, 100, 300, 1000, 2500])
    ax.set_xticklabels(["10", "30", "100", "300", "1 000", "2 500"])
    ax.minorticks_off()
    ax.grid(axis="y")
    ax.set_xlabel("Longueur de la critique en mots (échelle logarithmique)")
    ax.set_ylabel("Part des critiques de la classe (%)")
    ax.set_title("Les deux classes ont presque la même longueur", loc="left", fontweight="bold", pad=16)
    ax.legend(frameon=False, loc="upper right")
    return _sauver(fig, "01_longueurs_par_classe.png")


def figure_bruit(bruit: dict, notes: dict) -> plt.Figure:
    appliquer_style()
    lignes = [
        ("Balise <br />", bruit["balise_br"]["part_pct"]),
        ("Note écrite (ex. 8/10)", notes["part_pct"]),
        ("Adresse web", bruit["url"]["part_pct"]),
        ("Autre balise", bruit["autre_balise"]["part_pct"]),
        ("Entité HTML (ex. &amp;)", bruit["entite_html"]["part_pct"]),
    ][::-1]
    fig, ax = plt.subplots(figsize=(13, 6))
    noms, valeurs = zip(*lignes)
    barres = ax.barh(noms, valeurs, color="#2a78d6", height=0.58)
    for barre, valeur in zip(barres, valeurs):
        decimales = 1 if valeur >= 1 else 2
        ax.text(valeur + 0.8, barre.get_y() + barre.get_height() / 2, f"{fr(valeur, decimales)} %",
                va="center", color=ENCRE)
    ax.set_xlim(0, max(valeurs) * 1.14)
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Part des critiques concernées (%)")
    ax.set_title("La balise <br /> touche plus d'une critique sur deux", loc="left", fontweight="bold", pad=16)
    return _sauver(fig, "02_bruit_html_et_notes.png")


def figure_mots_frequents(frequents: dict) -> plt.Figure:
    appliquer_style()
    maximum = max(max(v.values()) for v in frequents.values())
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    for ax, classe in zip(axes, CLASSES):
        serie = pd.Series(frequents[classe]).sort_values()
        ax.barh(serie.index, serie.values, color=COULEURS[classe], height=0.66)
        ax.set_xlim(0, maximum * 1.05)
        ax.grid(axis="x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(LIBELLES[classe], loc="left", fontweight="bold")
        ax.set_xlabel("Occurrences pour 1 000 mots")
    fig.suptitle("« movie » et « film » dominent les deux classes", x=0.01, ha="left",
                 fontsize=22, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _sauver(fig, "03_mots_frequents_par_classe.png")


def figure_mots_associes(associes: dict) -> plt.Figure:
    appliquer_style()
    serie = pd.concat([pd.Series(associes["negative"]), pd.Series(associes["positive"])]).sort_values()
    fig, ax = plt.subplots(figsize=(13, 10))
    couleurs = [COULEURS["positive"] if v > 0 else COULEURS["negative"] for v in serie.values]
    ax.barh(serie.index, serie.values, color=couleurs, height=0.68)
    ax.axvline(0, color=AXE, linewidth=1)
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("log2(fréquence positive / fréquence négative)")
    ax.set_title("Les mots qui séparent le mieux les deux classes", loc="left", fontweight="bold", pad=34)
    ax.text(0, 1.012, f"Mots présents au moins {associes['min_occurrences']} fois.  Rouge : négatif.  Bleu : positif.",
            transform=ax.transAxes, color=ENCRE_2)
    return _sauver(fig, "04_mots_associes_par_classe.png")


def figure_notes(notes: dict) -> plt.Figure:
    appliquer_style()
    x = np.arange(11)
    largeur = 0.39
    fig, ax = plt.subplots(figsize=(13, 7))
    for decalage, classe in zip((-largeur / 2 - 0.012, largeur / 2 + 0.012), CLASSES):
        valeurs = [notes["critiques_par_note_entiere"][classe][str(v)] for v in x]
        ax.bar(x + decalage, valeurs, width=largeur, color=COULEURS[classe], label=LIBELLES[classe])
    ax.set_xticks(x)
    ax.grid(axis="y")
    ax.set_xlabel("Note écrite dans la critique, sur 10 (partie entière)")
    ax.set_ylabel("Nombre de critiques")
    ax.set_title(f"Une note nette suit l'étiquette dans {fr(notes['accord_pct'], 1)} % des cas",
                 loc="left", fontweight="bold", pad=38)
    ax.text(0, 1.018, "Note nette : toutes les notes de la critique sont ≤ 4, ou toutes sont ≥ 7.",
            transform=ax.transAxes, color=ENCRE_2)
    ax.legend(frameon=False, loc="upper center", ncols=2)
    return _sauver(fig, "05_notes_dans_le_texte.png")

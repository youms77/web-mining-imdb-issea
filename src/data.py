"""Chargement et dédoublonnage du corpus IMDB.

Exemple d'utilisation :
    from src.data import charger_brut, dedoublonner
    brut = charger_brut()
    corpus = dedoublonner(brut)  # mode retenu : "sans_ponctuation"
"""
from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
CHEMIN_BRUT = RACINE / "data" / "raw" / "IMDB Dataset.csv"
SHA256_BRUT = "DFC447764F82BE365FA9C2BEEF4E8DF89D3919E3DA95F5088004797D79695AA2"
COLONNES = ("review", "sentiment")
CLASSES = ("negative", "positive")

_BALISE_BR = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_ESPACES = re.compile(r"\s+")
_NON_ALPHANUM = re.compile(r"[\W_]+")  # garde les lettres accentuées et les chiffres


def empreinte_sha256(chemin: str | Path = CHEMIN_BRUT) -> str:
    """Calcule l'empreinte SHA256 d'un fichier, en majuscules."""
    calcul = hashlib.sha256()
    with Path(chemin).open("rb") as fichier:
        for bloc in iter(lambda: fichier.read(1024 * 1024), b""):
            calcul.update(bloc)
    return calcul.hexdigest().upper()


def charger_brut(chemin: str | Path = CHEMIN_BRUT, verifier: bool = True) -> pd.DataFrame:
    """Lit le CSV sans le modifier et ajoute `id_brut`, le numéro de ligne d'origine."""
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {chemin}")
    if verifier and empreinte_sha256(chemin) != SHA256_BRUT:
        raise ValueError("L'empreinte SHA256 du CSV ne correspond pas au fichier de référence.")

    # keep_default_na=False : un texte comme « NA » reste un texte, un vide reste visible.
    df = pd.read_csv(chemin, encoding="utf-8", dtype=str, keep_default_na=False)
    if tuple(df.columns) != COLONNES:
        raise ValueError(f"Colonnes inattendues : {list(df.columns)}")
    inconnues = set(df["sentiment"]) - set(CLASSES)
    if inconnues:
        raise ValueError(f"Étiquettes inattendues : {sorted(inconnues)}")
    return df.assign(id_brut=range(len(df)))


def cle_normalisee(texte: str) -> str:
    """Clé de comparaison : entités décodées, <br> retirés, minuscules, espaces réduits."""
    texte = html.unescape(texte)
    texte = _BALISE_BR.sub(" ", texte)
    return _ESPACES.sub(" ", texte.casefold()).strip()


def cle_sans_ponctuation(texte: str) -> str:
    """Clé la plus stricte : clé normalisée, sans ponctuation ni espaces."""
    return _NON_ALPHANUM.sub("", cle_normalisee(texte))


def diagnostiquer_doublons(df: pd.DataFrame, cles: pd.Series) -> dict[str, int]:
    """Compte les doublons et les groupes d'étiquettes contradictoires pour une clé donnée."""
    groupes = df["sentiment"].groupby(cles.to_numpy(), sort=False).agg(["size", "nunique"])
    contradictoires = groupes["nunique"] > 1
    return {
        "lignes": int(len(df)),
        "textes_uniques": int(len(groupes)),
        "lignes_en_trop": int(len(df) - len(groupes)),
        "groupes_repetes": int((groupes["size"] > 1).sum()),
        "groupes_contradictoires": int(contradictoires.sum()),
        "lignes_contradictoires": int(groupes.loc[contradictoires, "size"].sum()),
    }


def dedoublonner(df: pd.DataFrame, mode: str = "sans_ponctuation") -> pd.DataFrame:
    """Garde la première occurrence et retire tout groupe aux étiquettes contradictoires.

    Modes : "exact" (texte tel quel), "normalise" (`cle_normalisee`),
    "sans_ponctuation" (`cle_sans_ponctuation`, retenu car il retire aussi les copies
    qui ne diffèrent que par un espace ou des guillemets).
    """
    if mode == "exact":
        cles = df["review"]
    elif mode == "normalise":
        cles = df["review"].map(cle_normalisee)
    elif mode == "sans_ponctuation":
        cles = df["review"].map(cle_sans_ponctuation)
    else:
        raise ValueError("mode doit valoir 'exact', 'normalise' ou 'sans_ponctuation'.")

    nb_etiquettes = df["sentiment"].groupby(cles.to_numpy()).transform("nunique")
    coherent = nb_etiquettes.to_numpy() == 1
    premier = ~cles.duplicated(keep="first").to_numpy()
    return df.loc[coherent & premier].reset_index(drop=True)


def separer_entrainement_test(
    corpus: pd.DataFrame, part_test: float = 0.2, graine: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Séparation stratifiée du corpus dédoublonné. Le test ne sert qu'à la fin de la phase 5."""
    from sklearn.model_selection import train_test_split

    entrainement, test = train_test_split(
        corpus, test_size=part_test, stratify=corpus["sentiment"], random_state=graine
    )
    return entrainement.sort_index(), test.sort_index()

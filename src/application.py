"""Logique de l'application : contrôle du texte, sentiment, mots influents et thèmes.

Ce module ne dépend pas de Streamlit, ce qui permet de le tester seul. Il réutilise le
prétraitement de `src/` : l'application applique exactement les règles du modèle gelé.
Il n'importe ni torch, ni BERTopic, ni gensim. Tous les seuils sont lus dans `metrics.json`.

Exemple d'utilisation :
    from src.application import charger_artefacts, analyser
    artefacts = charger_artefacts()
    resultat = analyser("A long and honest movie review, in English...", artefacts)
    print(resultat["categorie"], resultat["probabilite_positive"])
"""
from __future__ import annotations

import traceback
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path

import joblib
import numpy as np

from src.models import preparer_texte
from src.preprocessing import RACINE, charger_ressources_b, pipeline_b

DOSSIER_DEPLOIEMENT = RACINE / "models" / "deploy"
CHEMIN_METRIQUES = RACINE / "reports" / "metrics.json"
FICHIER_SVM = "sentiment_svm_lineaire.joblib"
FICHIER_TOPICS = "topics_nmf_k8.joblib"
FICHIER_CONTENU = "contenu_app.json"

N_MOTS_INFLUENTS = 6
N_THEMES = 3
FENETRE_ALIGNEMENT = 12
_MORCEAU = re.compile(r"\S+")


class ErreurChargement(RuntimeError):
    """Un fichier nécessaire à l'application est absent ou illisible."""


@dataclass
class Artefacts:
    """Tout ce que l'application garde en mémoire après le premier chargement."""

    modele: object
    vectoriseur: object
    coefficients: np.ndarray
    constante: float
    vectoriseur_topics: object
    nmf: object
    mots_vides: frozenset
    unigrammes: frozenset
    metriques: dict
    contenu: dict
    fiches_topics: list


# --- Chargement --------------------------------------------------------------------------------

def charger_metriques(chemin: str | Path = CHEMIN_METRIQUES) -> dict:
    """Lit `metrics.json`, unique source des chiffres affichés."""
    chemin = Path(chemin)
    try:
        return json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, ValueError) as erreur:
        raise ErreurChargement(f"Le fichier de résultats est illisible : {chemin.name}.") from erreur


def charger_artefacts(
    dossier: str | Path = DOSSIER_DEPLOIEMENT, chemin_metriques: str | Path = CHEMIN_METRIQUES
) -> Artefacts:
    """Charge les modèles légers et prépare la lemmatisation, une seule fois."""
    dossier = Path(dossier)
    metriques = charger_metriques(chemin_metriques)
    try:
        modele = joblib.load(dossier / FICHIER_SVM)
        topics = joblib.load(dossier / FICHIER_TOPICS)
        contenu = json.loads((dossier / FICHIER_CONTENU).read_text(encoding="utf-8"))
        ressources_b = charger_ressources_b()
        pipeline_b("loading the language resources")  # évite cette attente au premier clic
    except (OSError, ValueError, KeyError, RuntimeError, EOFError, ModuleNotFoundError) as erreur:
        traceback.print_exc()  # cause réelle dans les journaux du serveur, le message utilisateur reste générique
        raise ErreurChargement(f"Un modèle n'a pas pu être chargé depuis {dossier.name}.") from erreur

    base = modele.calibrated_classifiers_[0].estimator
    vectoriseur = base.named_steps["vectoriseur"]
    classifieur = base.named_steps["classifieur"]
    return Artefacts(
        modele=modele,
        vectoriseur=vectoriseur,
        coefficients=np.asarray(classifieur.coef_).ravel(),
        constante=float(np.asarray(classifieur.intercept_).ravel()[0]),
        vectoriseur_topics=topics["vectoriseur"],
        nmf=topics["nmf"],
        mots_vides=frozenset(ressources_b["mots_vides"]),
        unigrammes=frozenset(t for t in vectoriseur.vocabulary_ if " " not in t),
        metriques=metriques,
        contenu=contenu,
        fiches_topics=metriques["topics"]["fiches_topics"]["nmf"],
    )


# --- Contrôle du texte -------------------------------------------------------------------------

def signaux_langue(jetons: list[str], mots_vides, unigrammes) -> tuple[float, float]:
    """Part de mots vides anglais, et part de mots connus (mots vides ou vocabulaire du modèle)."""
    if not jetons:
        return 0.0, 0.0
    vides = sum(j in mots_vides for j in jetons)
    connus = sum(j in mots_vides or j in unigrammes for j in jetons)
    return vides / len(jetons), connus / len(jetons)


def part_lettres_latines(texte: str) -> float | None:
    """Part des lettres de l'alphabet latin. None si le texte ne contient aucune lettre."""
    lettres = [c for c in texte if c.isalpha()]
    if not lettres:
        return None
    return sum(ord(c) < 0x250 for c in lettres) / len(lettres)


def controler_texte(texte: object, artefacts: Artefacts) -> dict:
    """Décide si le texte peut être analysé. `bloquant` empêche tout calcul de score."""
    seuils = artefacts.metriques["application"]["seuils_texte"]
    brut = texte.strip() if isinstance(texte, str) else ""
    if not brut:
        return _controle("vide", True, "Collez ou écrivez une critique en anglais avant de lancer l'analyse.")
    if len(brut) > seuils["caracteres_max"]:
        return _controle(
            "trop_long", True,
            f"Ce texte compte {fr_entier(len(brut))} caractères. La plus longue critique du corpus en compte "
            f"{fr_entier(seuils['caracteres_max'])} : raccourcissez-le pour rester dans ce que le modèle connaît.",
        )
    latines = part_lettres_latines(brut)
    if latines is not None and latines < seuils["part_lettres_latines_min"]:
        return _controle("non_anglais", True, _MESSAGE_NON_ANGLAIS)

    jetons = preparer_texte(brut).split()
    if len(jetons) < seuils["mots_min_analyse"]:
        return _controle(
            "trop_court", True,
            f"Texte trop court : {len(jetons)} mot(s) utile(s). Il en faut au moins "
            f"{seuils['mots_min_analyse']}, la longueur de la plus courte critique d'entraînement.",
            jetons,
        )
    vides, connus = signaux_langue(jetons, artefacts.mots_vides, artefacts.unigrammes)
    if vides < seuils["part_mots_vides_min"] and connus < seuils["part_mots_connus_min"]:
        return _controle("non_anglais", True, _MESSAGE_NON_ANGLAIS, jetons)
    if len(jetons) < seuils["mots_min_fiable"]:
        return _controle(
            "court", False,
            f"Texte court : {len(jetons)} mots utiles, sous le seuil de {seuils['mots_min_fiable']}. Le modèle "
            "est calibré sur des critiques entières : ce score n'est pas fiable, ne l'interprétez pas comme "
            "une probabilité.",
            jetons,
        )
    return _controle("ok", False, "", jetons)


_MESSAGE_NON_ANGLAIS = (
    "Ce texte ne semble pas écrit en anglais. Le modèle n'a appris que sur des critiques IMDB en anglais : "
    "il ne donnerait pas de résultat exploitable."
)


def _controle(code: str, bloquant: bool, message: str, jetons: list[str] | None = None) -> dict:
    return {"code": code, "bloquant": bloquant, "message": message, "jetons": jetons or []}


# --- Sentiment et mots influents ----------------------------------------------------------------

def categoriser(probabilite: float, seuil_bas: float, seuil_haut: float) -> str:
    """Même règle que `appliquer_zone` : les deux seuils sont hors de la zone d'incertitude."""
    if probabilite >= seuil_haut:
        return "positif"
    if probabilite <= seuil_bas:
        return "negatif"
    return "incertain"


def contributions_par_position(jetons: list[str], vectoriseur, coefficients: np.ndarray) -> np.ndarray:
    """Part du score SVM portée par chaque mot.

    Le poids d'un terme est réparti entre ses occurrences, et celui d'un bigramme entre ses deux
    mots. La somme vaut donc exactement la fonction de décision moins la constante.
    """
    sortie = np.zeros(len(jetons))
    if not jetons:
        return sortie
    matrice = vectoriseur.transform([" ".join(jetons)]).tocsr()
    valeurs = dict(zip(matrice.indices.tolist(), matrice.data.tolist()))
    occurrences: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    for i, mot in enumerate(jetons):
        occurrences[mot].append((i,))
        if i + 1 < len(jetons):
            occurrences[f"{mot} {jetons[i + 1]}"].append((i, i + 1))
    for terme, positions in occurrences.items():
        indice = vectoriseur.vocabulary_.get(terme)
        if indice is None or indice not in valeurs:
            continue
        part = valeurs[indice] * coefficients[indice] / (len(positions) * len(positions[0]))
        for groupe in positions:
            for i in groupe:
                sortie[i] += part
    return sortie


def mots_influents(jetons: list[str], contributions: np.ndarray, n: int = N_MOTS_INFLUENTS) -> dict:
    """Les mots qui poussent le plus vers positif et vers négatif."""
    cumul: dict[str, float] = defaultdict(float)
    for mot, valeur in zip(jetons, contributions):
        cumul[mot] += float(valeur)
    classes = sorted(cumul.items(), key=lambda paire: paire[1])
    return {
        "negatifs": [(m, v) for m, v in classes[:n] if v < 0],
        "positifs": [(m, v) for m, v in reversed(classes[-n:]) if v > 0],
    }


def segments_surlignes(texte: str, jetons: list[str], influents: dict) -> list[tuple[str, str | None]]:
    """Découpe le texte d'origine en morceaux marqués « pos », « neg » ou non marqués."""
    positifs = {m for m, _ in influents["positifs"]}
    negatifs = {m for m, _ in influents["negatifs"]}
    segments: list[tuple[str, str | None]] = []
    curseur, position = 0, 0
    for correspondance in _MORCEAU.finditer(texte):
        debut, fin = correspondance.span()
        if debut > curseur:
            segments.append((texte[curseur:debut], None))
        morceau = preparer_texte(texte[debut:fin]).split()
        nature = None
        if morceau:
            decalage = next(
                (d for d in range(FENETRE_ALIGNEMENT + 1) if jetons[position + d:position + d + len(morceau)] == morceau),
                None,
            )
            if decalage is not None:
                position += decalage + len(morceau)
                if any(m in negatifs for m in morceau):
                    nature = "neg"
                if any(m in positifs for m in morceau):
                    nature = "pos"
        segments.append((texte[debut:fin], nature))
        curseur = fin
    if curseur < len(texte):
        segments.append((texte[curseur:], None))
    return segments


def html_surligne(segments: list[tuple[str, str | None]]) -> str:
    """HTML du texte surligné. Le texte saisi est toujours échappé."""
    morceaux = []
    for morceau, nature in segments:
        contenu = escape(morceau)
        if nature == "pos":
            morceaux.append(f'<mark class="mot-pos" title="Pousse vers positif">{contenu}</mark>')
        elif nature == "neg":
            morceaux.append(f'<mark class="mot-neg" title="Pousse vers négatif">{contenu}</mark>')
        else:
            morceaux.append(contenu)
    return "".join(morceaux)


# --- Thèmes -------------------------------------------------------------------------------------

def themes_principaux(texte: str, artefacts: Artefacts, n: int = N_THEMES) -> list[dict]:
    """Les thèmes NMF les plus présents. Les parts sont des poids ramenés à 100 %."""
    texte_b = pipeline_b(texte)
    matrice = artefacts.vectoriseur_topics.transform([texte_b])
    if matrice.nnz == 0:
        return []
    poids = artefacts.nmf.transform(matrice)[0]
    total = float(poids.sum())
    if total <= 0:
        return []
    parts = poids / total
    fiches = {f["topic"]: f for f in artefacts.fiches_topics}
    return [
        {"topic": int(t), "nom": fiches[int(t)]["nom"], "part_pct": 100 * float(parts[t]),
         "mots": fiches[int(t)]["mots"][:5]}
        for t in np.argsort(-parts, kind="stable")[:n] if parts[t] > 0
    ]


# --- Analyse complète ---------------------------------------------------------------------------

def analyser(texte: object, artefacts: Artefacts) -> dict:
    """Contrôle le texte, puis calcule le sentiment, les mots influents et les thèmes."""
    controle = controler_texte(texte, artefacts)
    if controle["bloquant"]:
        return {"statut": "erreur", "code": controle["code"], "message": controle["message"]}

    brut = texte.strip()
    jetons = controle["jetons"]
    zone = artefacts.metriques["sentiment"]["choix_figes"]["zone_incertitude"]
    probabilite = float(artefacts.modele.predict_proba([" ".join(jetons)])[0, 1])
    contributions = contributions_par_position(jetons, artefacts.vectoriseur, artefacts.coefficients)
    influents = mots_influents(jetons, contributions)
    return {
        "statut": "ok",
        "code": controle["code"],
        "fiable": controle["code"] == "ok",
        "avertissement": controle["message"],
        "probabilite_positive": probabilite,
        "categorie": categoriser(probabilite, zone["seuil_bas"], zone["seuil_haut"]),
        "seuil_bas": zone["seuil_bas"],
        "seuil_haut": zone["seuil_haut"],
        "n_mots": len(jetons),
        "influents": influents,
        "html_surligne": html_surligne(segments_surlignes(brut, jetons, influents)),
        "themes": themes_principaux(brut, artefacts),
    }


# --- Format français ----------------------------------------------------------------------------

def fr_nombre(valeur: float, decimales: int = 2) -> str:
    """Nombre au format français : virgule décimale, espace insécable pour les milliers."""
    return f"{valeur:,.{decimales}f}".replace(",", " ").replace(".", ",")


def fr_entier(valeur: float) -> str:
    return fr_nombre(valeur, 0)


def fr_pct(proportion: float, decimales: int = 1) -> str:
    """Proportion (0,92) affichée en pourcentage (92,0 %)."""
    return f"{fr_nombre(100 * proportion, decimales)} %"

"""Prétraitement des critiques : un tronc commun et deux pipelines.

- Tronc commun : entités et balises HTML, URL, minuscules, contractions, espaces.
- Pipeline A (classification) : garde les négations et les notes du type « 8/10 ».
- Pipeline B (topics) : mots vides anglais et du domaine, lemmatisation, expressions de deux mots.

Ce module est importé par le notebook et par l'application : aucune copie de ces règles
ailleurs. Il n'utilise ni torch ni modèle lourd. NLTK n'est chargé que si un calcul
linguistique est demandé.

Exemple d'utilisation :
    from src.preprocessing import tronc_commun, pipeline_a, pipeline_b
    tronc_commun("I didn't like it.<br />Bad!")   # "i did not like it. bad!"
    pipeline_a("I didn't like it.<br />Bad!")     # "did not like it bad"
    pipeline_b("The special effects were stunning")  # "special_effect stunning"
"""
from __future__ import annotations

import os
import shutil
import tempfile
import html
import json
import re
from pathlib import Path
from typing import Iterable

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_NLTK = RACINE / "data" / "interim" / "nltk_data"
# Copie légère embarquée avec l'application : `data/interim/` n'existe pas au déploiement.
DOSSIER_NLTK_DEPLOIEMENT = RACINE / "models" / "deploy" / "nltk_data"
CHEMIN_RESSOURCES_B = RACINE / "src" / "ressources" / "pipeline_b.json"
RESSOURCES_NLTK = {
    "wordnet": "corpora/wordnet.zip",
    # Barre finale : NLTK lit alors l'archive .zip embarquée, sans dossier décompressé.
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng/",
    "stopwords": "corpora/stopwords",
}
# Seules ces deux ressources servent à lemmatiser. Les mots vides sont déjà dans pipeline_b.json.
RESSOURCES_LEMMATISATION = ("wordnet", "averaged_perceptron_tagger_eng")

# Motif à passer au vectoriseur : le texte du pipeline A est déjà découpé par des espaces.
MOTIF_VECTORISEUR_A = r"(?u)\S+"
JETON_EXCLAMATION = "__exclam__"
JETON_INTERROGATION = "__interro__"

# --- Tronc commun -------------------------------------------------------------------------

# Entité complète avec point-virgule. html.unescape seul convertirait « &not » en « ¬ ».
_ENTITE = re.compile(r"&(?:#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|[a-zA-Z][a-zA-Z0-9]{1,31});")

# Motif strict : seuls des noms de balises HTML reconnus, pour garder <grin>, <sigh> ou <sp?).
_NOMS_BALISES = (
    "a|abbr|b|big|blockquote|br|center|cite|code|div|em|font|h[1-6]|hr|i|img|li|ol|p|pre|"
    "s|small|span|strike|strong|sub|sup|table|td|th|tr|tt|u|ul"
)
# Après le nom, seuls de vrais attributs (nom=valeur) sont admis : « <i think so> » reste du texte.
_BALISE = re.compile(
    rf"</?(?:{_NOMS_BALISES})"
    r"(?:\s+[a-zA-Z:-]{1,30}\s*=\s*(?:\"[^\"<>]{0,300}\"|'[^'<>]{0,300}'|[^\s<>]{1,300}))*"
    r"\s*/?>",
    flags=re.IGNORECASE,
)
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"]+", flags=re.IGNORECASE)
_ESPACES = re.compile(r"\s+")

# Caractères Windows-1252 mal décodés observés dans le corpus (\x85, \x91, \x96, \x97...).
_TABLE_CARACTERES = str.maketrans({
    "\x85": " ... ", "\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "\x96": " - ", "\x97": " - ",
    "‘": "'", "’": "'", "ʼ": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "–": " - ", "—": " - ", "…": " ... ", "\xa0": " ",
})

_NEGATIONS_IRREGULIERES = {"can't": "can not", "won't": "will not", "shan't": "shall not", "ain't": "is not"}
# Négations écrites sans apostrophe, fréquentes dans le corpus (« dont », « cant », « didnt »).
_NEGATIONS_SANS_APOSTROPHE = {
    "dont": "do not", "didnt": "did not", "doesnt": "does not", "isnt": "is not", "wasnt": "was not",
    "werent": "were not", "arent": "are not", "cant": "can not", "cannot": "can not",
    "couldnt": "could not", "wouldnt": "would not", "shouldnt": "should not", "havent": "have not",
    "hasnt": "has not", "hadnt": "had not", "wont": "will not", "aint": "is not",
}
_RE_IRREGULIERES = re.compile(r"\b(?:can't|won't|shan't|ain't)\b")
_RE_SANS_APOSTROPHE = re.compile(r"\b(?:" + "|".join(_NEGATIONS_SANS_APOSTROPHE) + r")\b")
_RE_NT = re.compile(r"\b([a-z]+)n't\b")
_RE_RE = re.compile(r"\b([a-z]+)'re\b")
_RE_VE = re.compile(r"\b([a-z]+)'ve\b")
_RE_LL = re.compile(r"\b([a-z]+)'ll\b")
_RE_D = re.compile(r"\b(i|you|he|she|it|we|they|who|that|there)'d\b")
_RE_M = re.compile(r"\bi'm\b")
_RE_S_PRONOM = re.compile(r"\b(it|that|there|he|she|what|who|here|how|where)'s\b")
_RE_LETS = re.compile(r"\blet's\b")
_RE_POSSESSIF = re.compile(r"\b([^\W\d_]+)'s\b")


def _decoder_entite(correspondance: re.Match) -> str:
    return html.unescape(correspondance.group(0))


def nettoyer_html(texte: str) -> str:
    """Décode les entités complètes et remplace les balises reconnues par un espace."""
    return _BALISE.sub(" ", _ENTITE.sub(_decoder_entite, texte))


def retirer_urls(texte: str) -> str:
    """Remplace les adresses web par un espace."""
    return _URL.sub(" ", texte)


def developper_contractions(texte: str) -> str:
    """Développe les contractions d'un texte en minuscules : « didn't » devient « did not »."""
    texte = _RE_IRREGULIERES.sub(lambda m: _NEGATIONS_IRREGULIERES[m.group(0)], texte)
    texte = _RE_NT.sub(r"\1 not", texte)
    texte = _RE_SANS_APOSTROPHE.sub(lambda m: _NEGATIONS_SANS_APOSTROPHE[m.group(0)], texte)
    texte = _RE_RE.sub(r"\1 are", texte)
    texte = _RE_VE.sub(r"\1 have", texte)
    texte = _RE_LL.sub(r"\1 will", texte)
    texte = _RE_D.sub(r"\1 would", texte)
    texte = _RE_M.sub("i am", texte)
    texte = _RE_S_PRONOM.sub(r"\1 is", texte)
    texte = _RE_LETS.sub("let us", texte)
    return _RE_POSSESSIF.sub(r"\1", texte)  # « film's » devient « film »


def tronc_commun(texte: object, contractions: bool = True) -> str:
    """Nettoyage partagé par les deux pipelines. Une valeur non textuelle donne une chaîne vide."""
    if not isinstance(texte, str) or not texte:
        return ""
    texte = nettoyer_html(texte)
    texte = retirer_urls(texte)
    texte = texte.translate(_TABLE_CARACTERES).lower()
    if contractions:
        texte = developper_contractions(texte)
    return _ESPACES.sub(" ", texte).strip()


# --- Pipeline A : classification -----------------------------------------------------------

# Une note entière ou décimale sur 10 reste un seul jeton ; son effet est mesuré en phase 5.
_JETON_A = re.compile(
    r"__(?:exclam|interro)__"
    r"|(?<![\d/.,])\d{1,2}(?:[.,]\d{1,2})?/10(?![\d/])"
    r"|[^\W_]{2,}"
)
_EXCLAMATION = re.compile(r"!+")
_INTERROGATION = re.compile(r"\?+")


def marquer_ponctuation(texte: str) -> str:
    """Remplace chaque suite de « ! » ou de « ? » par un jeton que le vectoriseur conserve."""
    texte = _EXCLAMATION.sub(f" {JETON_EXCLAMATION} ", texte)
    return _INTERROGATION.sub(f" {JETON_INTERROGATION} ", texte)


# Même définition d'une note que l'exploration de la phase 1 (`src/exploration.py`).
MOTIF_NOTE = r"(?<![\d/])(?<!\d[.,])(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:/|\s+out\s+of\s+)\s*10(?![\d/]|[a-z])"
_NOTE_ECRITE = re.compile(MOTIF_NOTE, flags=re.IGNORECASE)


def retirer_notes(texte: str) -> str:
    """Retire d'une critique brute les notes sur 10 écrites en chiffres (« 8/10 », « 3 out of 10 »)."""
    return _NOTE_ECRITE.sub(" ", texte)


def tokeniser_a(texte: str) -> list[str]:
    """Découpe un texte du tronc commun en jetons d'au moins deux caractères, notes comprises."""
    return _JETON_A.findall(texte)


def pipeline_a(
    texte: object,
    ponctuation: bool = False,
    normalisation: str | None = None,
    contractions: bool = True,
) -> str:
    """Texte prêt pour la classification, en jetons séparés par des espaces.

    Les valeurs par défaut sont celles retenues par la validation croisée de la phase 2.
    `normalisation` vaut None, "lemme" ou "racine".
    """
    base = tronc_commun(texte, contractions=contractions)
    if ponctuation:
        base = marquer_ponctuation(base)
    jetons = tokeniser_a(base)
    if normalisation == "lemme":
        jetons = lemmatiser(jetons)
    elif normalisation == "racine":
        jetons = raciniser(jetons)
    elif normalisation is not None:
        raise ValueError("normalisation doit valoir None, 'lemme' ou 'racine'.")
    return " ".join(jetons)


# --- Ressources linguistiques --------------------------------------------------------------

_etat_nltk: dict = {}


def installer_ressources_nltk() -> None:
    """Télécharge les ressources NLTK dans data/interim/nltk_data (dossier non versionné)."""
    import nltk

    DOSSIER_NLTK.mkdir(parents=True, exist_ok=True)
    for ressource in RESSOURCES_NLTK:
        if not nltk.download(ressource, download_dir=str(DOSSIER_NLTK), quiet=True):
            raise RuntimeError(f"Téléchargement NLTK impossible : {ressource}")


def _dossier_nltk_sur(dossier: Path) -> Path:
    """Renvoie un dossier que NLTK accepte de lire.

    NLTK 3.10 refuse un dossier modifiable par un autre compte (cas du serveur Streamlit Community
    Cloud). Sous Linux, un dossier non privé est alors copié dans un dossier temporaire privé.
    """
    if os.name == "nt":
        return dossier
    info = dossier.stat()
    if info.st_uid == os.getuid() and not info.st_mode & 0o022:
        return dossier
    prive = Path(tempfile.mkdtemp(prefix="nltk_data_")) / "nltk_data"
    shutil.copytree(dossier, prive)
    for chemin in [prive, *prive.rglob("*")]:
        chemin.chmod(0o700 if chemin.is_dir() else 0o600)
    return prive


def _nltk_pret() -> dict:
    """Charge une seule fois le lemmatiseur, le raciniseur et l'étiqueteur grammatical."""
    if _etat_nltk:
        return _etat_nltk
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tag import PerceptronTagger

    for dossier in (DOSSIER_NLTK_DEPLOIEMENT, DOSSIER_NLTK):
        if dossier.exists():
            dossier = _dossier_nltk_sur(dossier)
            if str(dossier) not in nltk.data.path:
                nltk.data.path.insert(0, str(dossier))
    try:
        for ressource in RESSOURCES_LEMMATISATION:
            nltk.data.find(RESSOURCES_NLTK[ressource])
    except LookupError as erreur:
        raise RuntimeError(
            "Ressources NLTK absentes. Lancez : "
            "python -c \"from src.preprocessing import installer_ressources_nltk as f; f()\""
        ) from erreur
    lemmatiseur = WordNetLemmatizer()
    lemmatiseur.lemmatize("films")  # force le chargement de WordNet
    _etat_nltk.update(
        lemmatiseur=lemmatiseur,
        raciniseur=PorterStemmer(),
        etiqueteur=PerceptronTagger(lang="eng"),
        cache_lemmes={},
        cache_racines={},
    )
    return _etat_nltk


def _categorie_wordnet(etiquette: str) -> str | None:
    """Traduit une étiquette Penn Treebank. None : le mot ne doit pas être lemmatisé."""
    if etiquette.startswith("NNP"):
        return None  # un nom propre reste tel quel
    return {"J": "a", "V": "v", "N": "n", "R": "r"}.get(etiquette[:1])


def lemmatiser(jetons: list[str]) -> list[str]:
    """Lemmatisation WordNet guidée par la catégorie grammaticale de chaque mot.

    Les mots-outils ne passent pas par WordNet : sans cela « his » deviendrait « hi ».
    Les jetons spéciaux et les nombres sont conservés.
    """
    if not jetons:
        return []
    outils = _nltk_pret()
    cache, lemmatiseur = outils["cache_lemmes"], outils["lemmatiseur"]
    sortie = []
    for mot, etiquette in outils["etiqueteur"].tag(jetons):
        categorie = _categorie_wordnet(etiquette)
        if categorie is None or not mot.isalpha():
            sortie.append(mot)
            continue
        cle = (mot, categorie)
        if cle not in cache:
            cache[cle] = lemmatiseur.lemmatize(mot, categorie)
        sortie.append(cache[cle])
    return sortie


def raciniser(jetons: list[str]) -> list[str]:
    """Racinisation de Porter. Les jetons spéciaux et les nombres sont conservés."""
    if not jetons:
        return []
    outils = _nltk_pret()
    cache, raciniseur = outils["cache_racines"], outils["raciniseur"]
    sortie = []
    for mot in jetons:
        if not mot.isalpha():
            sortie.append(mot)
            continue
        if mot not in cache:
            cache[mot] = raciniseur.stem(mot)
        sortie.append(cache[mot])
    return sortie


# --- Pipeline B : topics -------------------------------------------------------------------

_MOT_B = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
_LONGUEUR_MIN_B = 3
_ressources_b: dict | None = None


def tokeniser_b(texte: str) -> list[str]:
    """Découpe un texte du tronc commun en mots alphabétiques, sans nombres."""
    return _MOT_B.findall(texte)


def charger_ressources_b(chemin: str | Path = CHEMIN_RESSOURCES_B) -> dict:
    """Lit les mots vides et les expressions retenus, calculés une fois sur le corpus."""
    global _ressources_b
    chemin = Path(chemin)
    if _ressources_b is not None and chemin == CHEMIN_RESSOURCES_B:
        return _ressources_b
    if not chemin.exists():
        raise FileNotFoundError(f"Ressources du pipeline B introuvables : {chemin}")
    brut = json.loads(chemin.read_text(encoding="utf-8"))
    ressources = {
        "mots_vides": frozenset(brut["mots_vides_anglais"]),
        "mots_domaine": frozenset(brut["mots_vides_domaine"]),
        "bigrammes": frozenset(tuple(b.split("_")) for b in brut["bigrammes"]),
    }
    if chemin == CHEMIN_RESSOURCES_B:
        _ressources_b = ressources
    return ressources


def filtrer_b(formes: list[str], lemmes: list[str], mots_vides: Iterable[str]) -> list[str | None]:
    """Remplace par None les mots vides et les mots courts, pour garder trace des coupures."""
    return [
        None if (f in mots_vides or l in mots_vides or len(l) < _LONGUEUR_MIN_B) else l
        for f, l in zip(formes, lemmes)
    ]


def fusionner_bigrammes(sequence: list[str | None], bigrammes: Iterable[tuple[str, str]]) -> list[str]:
    """Relie deux mots réellement voisins s'ils forment une expression : special_effect."""
    sortie: list[str] = []
    i = 0
    while i < len(sequence):
        mot = sequence[i]
        if mot is None:
            i += 1
            continue
        suivant = sequence[i + 1] if i + 1 < len(sequence) else None
        if suivant is not None and (mot, suivant) in bigrammes:
            sortie.append(f"{mot}_{suivant}")
            i += 2
        else:
            sortie.append(mot)
            i += 1
    return sortie


def pipeline_b(texte: object, ressources: dict | None = None) -> str:
    """Texte prêt pour les topics : lemmes utiles et expressions, séparés par des espaces."""
    if ressources is None:
        ressources = charger_ressources_b()
    formes = tokeniser_b(tronc_commun(texte))
    if not formes:
        return ""
    # L'étiqueteur a besoin de la phrase entière : les mots vides sont retirés après lui.
    lemmes = lemmatiser(formes)
    sequence = filtrer_b(formes, lemmes, ressources["mots_vides"])
    # Les mots du domaine sont comparés au lemme : « movies » et « movie » partent ensemble.
    sequence = [None if m in ressources["mots_domaine"] else m for m in sequence]
    return " ".join(fusionner_bigrammes(sequence, ressources["bigrammes"]))

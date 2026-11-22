"""Tests du prétraitement : cas limites exigés par le cahier et pièges observés dans le corpus.

Lancement : .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_preprocessing.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from src import preprocessing as pp
from src.preprocessing import (
    fusionner_bigrammes,
    lemmatiser,
    marquer_ponctuation,
    nettoyer_html,
    pipeline_a,
    pipeline_b,
    raciniser,
    tokeniser_a,
    tronc_commun,
)

RESSOURCES_TEST = {
    "mots_vides": frozenset({"the", "was", "were", "be", "not", "a", "of", "it", "is", "and"}),
    "mots_domaine": frozenset({"movie", "film", "good"}),
    "bigrammes": frozenset({("special", "effect"), ("high", "school")}),
}


@pytest.fixture(scope="session", autouse=True)
def ressources_nltk():
    """Installe les ressources NLTK au premier lancement dans un environnement neuf."""
    try:
        pp._nltk_pret()
    except RuntimeError:
        pp.installer_ressources_nltk()
        pp._nltk_pret()


# --- Cas limites exigés ---------------------------------------------------------------------

@pytest.mark.parametrize("balise", ["<br />", "<br>", "<BR/>", "<i></i>", "<p>", "<hr>", "<br /><br />"])
def test_balise_seule_donne_un_texte_vide(balise):
    assert tronc_commun(balise) == ""
    assert pipeline_a(balise) == ""
    assert pipeline_b(balise, RESSOURCES_TEST) == ""


@pytest.mark.parametrize("vide", ["", "   ", "\n\t ", None, float("nan"), 42])
def test_texte_vide_ou_non_textuel(vide):
    assert tronc_commun(vide) == ""
    assert pipeline_a(vide) == ""
    assert pipeline_b(vide, RESSOURCES_TEST) == ""


@pytest.mark.parametrize("texte, attendu", [
    ("I did not like it", "did not like"),
    ("This is not good", "not good"),
    ("no fun, never again", "no fun never"),
])
def test_negation_conservee_dans_a(texte, attendu):
    assert attendu in pipeline_a(texte)


def test_did_not_survit_au_pipeline_a():
    for texte in ("I did not enjoy it.", "I didn't enjoy it.", "I DIDN'T enjoy it.", "I didn`t enjoy it.",
                  "I didn´t enjoy it.", "I didn’t enjoy it.", "I didnt enjoy it."):
        assert "did not enjoy" in pipeline_a(texte), texte


def test_negation_retiree_dans_b():
    assert "not" not in pipeline_b("It was not a story", RESSOURCES_TEST).split()


@pytest.mark.parametrize("texte, attendu", [
    ("didn't", "did not"), ("don't", "do not"), ("doesn't", "does not"), ("isn't", "is not"),
    ("wasn't", "was not"), ("can't", "can not"), ("cannot", "can not"), ("won't", "will not"),
    ("shan't", "shall not"), ("ain't", "is not"), ("couldn't", "could not"), ("wouldn't", "would not"),
    ("dont", "do not"), ("cant", "can not"), ("it's", "it is"), ("that's", "that is"),
    ("i'm", "i am"), ("you're", "you are"), ("we've", "we have"), ("they'll", "they will"),
    ("i'd", "i would"), ("let's", "let us"), ("film's", "film"), ("wouldn't've", "would not have"),
])
def test_contractions(texte, attendu):
    assert tronc_commun(texte) == attendu


def test_emoji_ne_casse_rien():
    texte = "I loved it 😍🎬 best movie!"
    assert "😍" in tronc_commun(texte)
    assert pipeline_a(texte) == "loved it best movie"
    assert "😍" not in pipeline_b(texte, RESSOURCES_TEST)
    assert pipeline_a("😍😍😍") == ""
    assert pipeline_b("😍😍😍", RESSOURCES_TEST) == ""


def test_texte_tres_long():
    phrase = "I didn't like this movie.<br />It wasn't good & the plot was weak! "
    texte = phrase * 30_000  # environ 2 millions de caractères
    debut = time.perf_counter()
    resultat = pipeline_a(texte)
    assert time.perf_counter() - debut < 20
    assert resultat.split().count("not") == 60_000
    assert "br" not in resultat.split()
    assert pipeline_b(phrase * 2_000, RESSOURCES_TEST).split().count("plot") == 2_000


@pytest.mark.parametrize("piege", ["<" * 200_000, "&" * 200_000, "a" * 1_000_000, "<br " * 50_000, "'" * 200_000])
def test_entrees_pathologiques_restent_rapides(piege):
    debut = time.perf_counter()
    tronc_commun(piege)
    pipeline_a(piege)
    assert time.perf_counter() - debut < 5


# --- Pièges observés dans le corpus ----------------------------------------------------------

@pytest.mark.parametrize("fausse", ["<grin>", "<sigh>", "<sp?) classic"])
def test_fausses_balises_gardent_leur_mot(fausse):
    mot = fausse.strip("<").split(">")[0].split("?")[0]
    assert mot in pipeline_a(f"well {fausse} indeed").split()


def test_motif_strict_ne_mange_pas_le_texte():
    texte = "less < than 30 minutes of watching and x > y"
    assert "than 30 minutes of watching" in tronc_commun(texte)


def test_balises_avec_attributs_et_casse():
    assert tronc_commun('<A HREF="x.html">lien</A> <font color=red>rouge</font>') == "lien rouge"


def test_nom_de_balise_suivi_de_mots_reste_du_texte():
    assert "think so" in tronc_commun("well <i think so> indeed")


def test_entites_html():
    assert tronc_commun("Tom &amp; Jerry &#233;t&eacute;") == "tom & jerry été"
    assert tronc_commun("&lt;br /&gt;fin") == "fin"


def test_entite_sans_point_virgule_inchangee():
    # html.unescape seul produirait « rock¬ » et ferait disparaître le mot.
    assert nettoyer_html("rock&not roll") == "rock&not roll"
    assert "not" in pipeline_a("rock&not roll").split()


def test_caracteres_windows_1252():
    assert tronc_commun("great\x96really\x85 it\x92s") == "great - really ... it is"
    assert tronc_commun("didn\x92t") == "did not"


@pytest.mark.parametrize("url", ["http://www.imdb.com/title/tt1/", "https://a.org/x?y=1", "www.nixflix.com"])
def test_url_retirees(url):
    assert tronc_commun(f"voir {url} ensuite") == "voir ensuite"


def test_minuscules_et_espaces():
    assert tronc_commun("  UN   Grand\n\nFILM  ") == "un grand film"


@pytest.mark.parametrize("note", ["8/10", "10/10", "1/10", "7.5/10"])
def test_notes_conservees_dans_a(note):
    assert note in pipeline_a(f"I give it {note} overall").split()


def test_date_ne_devient_pas_une_note():
    assert "2006/10" not in tokeniser_a("released 2006/10/02")


def test_ponctuation_expressive():
    assert marquer_ponctuation("wow!!! really?") == f"wow {pp.JETON_EXCLAMATION}  really {pp.JETON_INTERROGATION} "
    assert pp.JETON_EXCLAMATION in pipeline_a("wow!!!", ponctuation=True).split()
    assert pp.JETON_EXCLAMATION not in pipeline_a("wow!!!", ponctuation=False).split()


def test_normalisation_invalide():
    with pytest.raises(ValueError):
        pipeline_a("texte", normalisation="autre")


# --- Lemmatisation et racinisation -----------------------------------------------------------

def test_lemmatisation_selon_la_categorie():
    assert lemmatiser(["the", "movies", "were", "better"]) == ["the", "movie", "be", "good"]
    assert lemmatiser(["he", "was", "studying"]) == ["he", "be", "study"]


def test_mots_outils_non_abimes():
    # Sans étiquette grammaticale, WordNet donne « hi » pour « his » et « wa » pour « was ».
    assert lemmatiser(["his", "film", "was", "as", "long"])[0] == "his"
    assert "wa" not in lemmatiser(["it", "was", "long"])


def test_racinisation_produit_des_formes_tronquees():
    assert raciniser(["studies", "movies"]) == ["studi", "movi"]


def test_jetons_speciaux_preserves():
    assert raciniser([pp.JETON_EXCLAMATION, "8/10"]) == [pp.JETON_EXCLAMATION, "8/10"]
    assert pp.JETON_EXCLAMATION in lemmatiser(["wow", pp.JETON_EXCLAMATION])
    assert lemmatiser([]) == [] and raciniser([]) == []


# --- Pipeline B ---------------------------------------------------------------------------------

def test_b_retire_mots_vides_et_domaine():
    sortie = pipeline_b("The movies were good and the story was strong", RESSOURCES_TEST).split()
    assert "movie" not in sortie and "good" not in sortie and "the" not in sortie
    assert "story" in sortie


def test_b_forme_les_expressions():
    assert "special_effect" in pipeline_b("The special effects were strong", RESSOURCES_TEST).split()


def test_b_ne_relie_pas_des_mots_separes_par_un_mot_vide():
    assert fusionner_bigrammes(["special", None, "effect"], RESSOURCES_TEST["bigrammes"]) == ["special", "effect"]


def test_b_ignore_les_nombres():
    assert "2006" not in pipeline_b("story from 2006", RESSOURCES_TEST)


def test_ressources_b_du_projet():
    ressources = pp.charger_ressources_b()
    assert {"movie", "film"} <= ressources["mots_domaine"]
    assert "not" in ressources["mots_vides"]
    assert not {"story", "character", "plot"} & (ressources["mots_vides"] | ressources["mots_domaine"])
    assert "special_effect" in pipeline_b("The special effects were impressive").split()


def test_reglages_par_defaut_conformes_a_la_validation_croisee():
    metrics = json.loads((pp.RACINE / "reports" / "metrics.json").read_text(encoding="utf-8"))
    retenus = metrics["pretraitement"]["validation_croisee_a"]["reglages_retenus"]
    assert pipeline_a("Wow! Movies?") == pipeline_a("Wow! Movies?", **retenus)


def test_ressources_b_absentes(tmp_path):
    with pytest.raises(FileNotFoundError):
        pp.charger_ressources_b(tmp_path / "absent.json")


# --- Déploiement -------------------------------------------------------------------------------

def test_import_leger_sans_torch_ni_nltk():
    code = (
        "import sys; import src.preprocessing as p; p.tronc_commun('x'); p.pipeline_a('x');"
        "print(any(m in sys.modules for m in ('torch', 'transformers', 'nltk', 'sklearn')))"
    )
    sortie = subprocess.run([sys.executable, "-c", code], cwd=pp.RACINE, capture_output=True, text=True, check=True)
    assert sortie.stdout.strip() == "False"

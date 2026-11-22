"""Tests de la phase 6 : nettoyage léger, découpage, échantillon, règle de choix, extension.

Lancement : .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_bertopic.py
Ces tests ne chargent pas le modèle d'embeddings. Ceux sur les caches sont ignorés s'ils sont absents.
"""
from __future__ import annotations

import ast
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from src import bertopic_model as bm
from src import bertopic_phrases as bp
from src import features as f
from src.preprocessing import RACINE


# --- Nettoyage léger -----------------------------------------------------------------------------

def test_nettoyage_leger_garde_mots_vides_casse_et_ponctuation():
    assert bm.nettoyer_leger("The Movie was NOT good, was it?") == "The Movie was NOT good, was it?"


def test_nettoyage_leger_retire_html_url_et_notes():
    texte = bm.nettoyer_leger("Great film.<br /><br />I give it 8/10. See http://example.com now &amp; later")
    assert "<br" not in texte and "8/10" not in texte and "http" not in texte
    assert "&" in texte and "\n" in texte


@pytest.mark.parametrize("valeur", [None, 3, ""])
def test_nettoyage_leger_valeur_non_textuelle(valeur):
    assert bm.nettoyer_leger(valeur) == ""


# --- Découpage ---------------------------------------------------------------------------------

def test_decoupage_phrases_simple():
    assert bm.decouper_phrases("Good actors. But the story goes nowhere!") == [
        "Good actors.", "But the story goes nowhere!"]


def test_decoupage_respecte_abreviations_et_initiales():
    assert bm.decouper_phrases("Mr. Smith met J. Doe. They left.") == ["Mr. Smith met J. Doe.", "They left."]


def test_decoupage_garde_les_guillemets_et_les_lignes():
    assert bm.decouper_phrases('"Really?" she said.\nThen it ended') == ['"Really?" she said.', "Then it ended"]


def test_decoupage_ne_perd_aucun_mot():
    texte = "One two. Three four! Five six? Seven eight"
    assert " ".join(bm.decouper_phrases(texte)).split() == texte.split()


def test_composer_blocs_respecte_le_maximum():
    blocs = bm.composer_blocs([3, 3, 3, 5, 1], 6)
    assert blocs == [(0, 2), (2, 3), (3, 5)]
    assert blocs[0][0] == 0 and blocs[-1][1] == 5


def test_composer_blocs_phrase_trop_longue_seule():
    assert bm.composer_blocs([2, 10, 2], 5) == [(0, 1), (1, 2), (2, 3)]


def test_composer_blocs_maximum_invalide():
    with pytest.raises(ValueError):
        bm.composer_blocs([1], 0)


def test_decouper_jetons_sans_perte():
    ids = list(range(23))
    tranches = bm.decouper_jetons(ids, 5)
    assert max(len(t) for t in tranches) == 5 and [x for t in tranches for x in t] == ids


# --- Agrégation ----------------------------------------------------------------------------------

def test_agregation_ponderee_et_normee():
    v = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    sortie = bm.agreger_blocs(v, np.array([0, 0, 1]), np.array([3.0, 1.0, 2.0], dtype=np.float32), 2)
    attendu = np.array([3.0, 1.0]) / np.sqrt(10)
    assert sortie[0] == pytest.approx(attendu, abs=1e-6)
    assert sortie[1] == pytest.approx([1.0, 0.0])
    assert np.linalg.norm(sortie, axis=1) == pytest.approx([1.0, 1.0])


def test_agregation_refuse_une_critique_sans_bloc():
    v = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        bm.agreger_blocs(v, np.array([0, 0]), np.ones(2, dtype=np.float32), 2)


# --- Échantillon --------------------------------------------------------------------------------

def test_repartir_quotas_somme_exacte():
    quotas = bm.repartir_quotas(np.array([10, 20, 30, 41]), 17)
    assert quotas.sum() == 17 and (quotas <= np.array([10, 20, 30, 41])).all()


def test_repartir_quotas_trop_grand():
    with pytest.raises(ValueError):
        bm.repartir_quotas(np.array([1, 2]), 4)


def _table(n=2000):
    g = np.random.default_rng(0)
    return pd.DataFrame({"y": g.integers(0, 2, n), "longueur": g.integers(10, 900, n)})


def test_tirage_equilibre_sans_doublon_et_reproductible():
    table = _table()
    a, b = bm.tirer_strates(table, 400), bm.tirer_strates(table, 400)
    assert np.array_equal(a, b) and len(set(a.tolist())) == 400
    assert table.iloc[a]["y"].sum() == 200


def test_tirage_garde_les_quartiles_de_longueur():
    table = _table(8000)
    choisies = table.iloc[bm.tirer_strates(table, 2000)]
    bornes = table["longueur"].quantile([0.25, 0.5, 0.75]).to_numpy()
    parts = np.histogram(choisies["longueur"], [-np.inf, *bornes, np.inf])[0] / len(choisies)
    assert parts == pytest.approx([0.25] * 4, abs=0.02)


def test_tirage_refuse_un_effectif_impair():
    with pytest.raises(ValueError):
        bm.tirer_strates(_table(), 101)


# --- Règle de choix de HDBSCAN ---------------------------------------------------------------------

def _ligne(taille, n, hors, div, cv, grand=0.2):
    return {"taille_min": taille, "n_topics": n, "part_hors_topic": hors, "diversite": div,
            "c_v_entrainement": cv, "plus_grand_topic": grand}


def test_regle_prefere_moins_de_hors_topic_dans_la_tolerance():
    lignes = [_ligne(15, 30, 0.5, 0.8, 0.500), _ligne(25, 20, 0.3, 0.8, 0.495), _ligne(40, 12, 0.2, 0.8, 0.480)]
    assert bm.choisir_taille_min(lignes)["taille_min"] == 25


def test_regle_ecarte_diversite_et_trop_peu_de_topics():
    lignes = [_ligne(15, 30, 0.1, 0.6, 0.9), _ligne(25, 3, 0.1, 0.9, 0.9), _ligne(40, 12, 0.4, 0.8, 0.4)]
    assert bm.choisir_taille_min(lignes)["taille_min"] == 40


def test_regle_egalite_prend_la_plus_grande_taille():
    lignes = [_ligne(15, 14, 0.3, 0.8, 0.5), _ligne(25, 13, 0.3, 0.8, 0.5)]
    assert bm.choisir_taille_min(lignes)["taille_min"] == 25


def test_regle_ecarte_un_topic_qui_absorbe_le_corpus():
    lignes = [_ligne(25, 14, 0.05, 0.9, 0.6, grand=0.93), _ligne(15, 36, 0.58, 0.87, 0.46, grand=0.10)]
    assert bm.choisir_taille_min(lignes)["taille_min"] == 15


def test_regle_sans_config_admissible():
    with pytest.raises(RuntimeError):
        bm.choisir_taille_min([_ligne(15, 2, 0.1, 0.1, 0.5)])


def test_reduction_precalculee_verifie_la_taille():
    r = bm.ReductionPrecalculee(np.zeros((3, 2)))
    assert r.fit(np.zeros((3, 5))).transform(np.zeros((3, 5))).shape == (3, 2)
    with pytest.raises(ValueError):
        r.fit(np.zeros((4, 5)))


def test_bertopic_garde_les_bigrammes_et_ignore_les_termes_absents():
    g = np.random.default_rng(0)
    centres = np.array([[8.0, 0, 0, 0, 0], [0, 8.0, 0, 0, 0]])
    groupes = np.repeat([0, 1], 30)
    coordonnees = (centres[groupes] + g.normal(0, 0.1, (60, 5))).astype(np.float32)
    documents = ["special_effect monster"] * 30 + ["love_story romance"] * 30
    termes = ["absent_terme", "love_story", "monster", "romance", "special_effect"]
    modele, sujets = bm.ajuster_bertopic(documents, coordonnees, coordonnees, termes, 5)
    assert set(sujets.tolist()) == {0, 1}
    mots = {m for t in bm.mots_topics(modele) for m in t}
    assert {"special_effect", "love_story"} <= mots and "absent_terme" not in mots


# --- Extension exploratoire ------------------------------------------------------------------------

def test_categories_gardent_les_bornes_en_zone_incertaine():
    assert bp.categoriser(np.array([0.1, 0.28, 0.5, 0.72, 0.9]), 0.28, 0.72).tolist() == [0, 1, 1, 1, 2]


def test_accord_vader_calcul_a_la_main():
    proba = np.array([0.9, 0.8, 0.2, 0.1, 0.6, 0.4])
    vader = np.array([0.5, -0.5, -0.5, -0.6, 0.0, 0.01])
    r = bp.mesurer_accord_vader(proba, vader, 0.28, 0.72)
    assert r["part_vader_neutre"] == pytest.approx(2 / 6)
    assert r["accord_brut_vader_non_neutre"] == pytest.approx(3 / 4)
    assert r["kappa_cohen_vader_non_neutre"] == pytest.approx(cohen_kappa_score([1, 0, 0, 0], [1, 1, 0, 0]))
    assert "exactitude" not in " ".join(r)


def test_part_critiques_avec_phrase():
    table = pd.DataFrame({
        "id_brut": [1, 1, 2, 2, 3], "y_critique": [1, 1, 1, 1, 0],
        "p": [0.9, 0.1, 0.8, 0.7, 0.9],
    })
    assert bp.part_critiques_avec_phrase(table, "p", 1, "negative", 0.28, 0.72) == pytest.approx(0.5)
    assert bp.part_critiques_avec_phrase(table, "p", 0, "positive", 0.28, 0.72) == pytest.approx(1.0)


def test_ic_grappes_encadre_la_proportion():
    g = np.random.default_rng(0)
    totaux = np.full(200, 5)
    comptes = g.binomial(5, 0.3, 200)
    tirages = np.vstack([np.bincount(g.integers(0, 200, 200), minlength=200) for _ in range(300)])
    bas, haut = bp._ic_grappes(comptes, totaux, tirages)
    assert bas < 100 * comptes.sum() / totaux.sum() < haut


# --- Garde-fous du projet --------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["bertopic_model", "bertopic_phrases"])
def test_modules_n_ouvrent_pas_le_test(module):
    arbre = ast.parse((RACINE / "src" / f"{module}.py").read_text(encoding="utf-8"))
    importes = set()
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Import):
            importes |= {a.name for a in noeud.names}
        elif isinstance(noeud, ast.ImportFrom):
            importes |= {f"{noeud.module}.{a.name}" for a in noeud.names}
    assert not {i for i in importes if "sentiment_test" in i}


@pytest.mark.parametrize("fichier", ["requirements.txt", "app/requirements.txt"])
def test_requirements_socle_sans_paquet_lourd(fichier):
    chemin = RACINE / fichier
    if not chemin.exists():
        pytest.skip("Fichier créé dans une phase ultérieure.")
    contenu = chemin.read_text(encoding="utf-8").lower()
    for paquet in ("torch", "bertopic", "transformers", "umap", "hdbscan"):
        assert paquet not in contenu


def test_importer_le_module_ne_charge_pas_torch():
    code = "import sys; import src.bertopic_model; print('torch' in sys.modules)"
    sortie = subprocess.run([sys.executable, "-c", code], cwd=RACINE, capture_output=True, text=True, check=True)
    assert sortie.stdout.strip() == "False"


@pytest.mark.skipif(not bm.CHEMIN_ECHANTILLON.exists(), reason="Échantillon absent.")
def test_echantillon_reel_hors_test_et_equilibre():
    e = pd.read_parquet(bm.CHEMIN_ECHANTILLON)
    _, id_test = f.charger_index()
    assert len(e) == bm.N_CRITIQUES and e["id_brut"].is_unique
    assert not set(e["id_brut"]) & set(id_test.tolist())
    assert e["y"].sum() == bm.N_CRITIQUES // 2
    assert e["pour_phrases"].sum() == bm.N_CRITIQUES_PHRASES
    assert e.loc[e["pour_phrases"], "y"].sum() == bm.N_CRITIQUES_PHRASES // 2

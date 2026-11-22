"""Tests du topic modeling descriptif et de la comparaison par polarité.

Lancement : .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_topics.py
Les tests sur le vrai corpus sont ignorés si les caches de data/interim sont absents.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src import topics as tp
from src.preprocessing import RACINE

CACHES_PRESENTS = (tp.DOSSIER_PHASE3 / "topics_bow.npz").exists()

# Petit corpus à deux thèmes nets : cuisine et espace.
TERMES = ["pain", "four", "sel", "fusee", "orbite", "lune"]
BOW = sparse.csr_matrix(np.array(
    [[5, 4, 3, 0, 0, 0], [4, 5, 2, 0, 0, 0], [6, 3, 4, 0, 0, 0], [3, 4, 5, 0, 0, 0],
     [0, 0, 0, 5, 4, 3], [0, 0, 0, 4, 5, 2], [0, 0, 0, 6, 3, 4], [0, 0, 0, 3, 4, 5]] * 5,
    dtype=np.int64,
))


# --- Mots principaux, diversité, normalisation -------------------------------------------------

def test_mots_principaux_tries_par_poids():
    composantes = np.array([[0.1, 0.7, 0.2], [0.5, 0.2, 0.3]])
    assert tp.mots_principaux(composantes, ["a", "b", "c"], n=2) == [["b", "c"], ["a", "c"]]


def test_diversite_bornes():
    assert tp.diversite([["a", "b"], ["c", "d"]]) == 1.0
    assert tp.diversite([["a", "b"], ["a", "b"]]) == 0.5
    assert tp.diversite([]) == 0.0


def test_normaliser_lignes_somme_a_un_meme_pour_une_ligne_nulle():
    sortie = tp.normaliser_lignes(np.array([[2.0, 2.0], [0.0, 0.0], [1.0, 3.0]]))
    assert np.allclose(sortie.sum(axis=1), 1)
    assert np.allclose(sortie[1], [0.5, 0.5])
    assert np.allclose(sortie[2], [0.25, 0.75])


# --- Modèles ------------------------------------------------------------------------------------

def test_lda_reproductible_avec_la_meme_graine():
    a = tp.ajuster_lda(BOW, TERMES, k=2, graine=42, passes=3)
    b = tp.ajuster_lda(BOW, TERMES, k=2, graine=42, passes=3)
    assert np.array_equal(a.get_topics(), b.get_topics())
    assert np.array_equal(tp.theta_lda(a, BOW), tp.theta_lda(b, BOW))


def test_lda_retrouve_deux_themes_et_theta_somme_a_un():
    modele = tp.ajuster_lda(BOW, TERMES, k=2, passes=10)
    mots = [set(t) for t in tp.mots_principaux(modele.get_topics(), TERMES, n=3)]
    assert {"pain", "four", "sel"} in mots and {"fusee", "orbite", "lune"} in mots
    theta = tp.theta_lda(modele, BOW)
    assert theta.shape == (BOW.shape[0], 2)
    assert np.allclose(theta.sum(axis=1), 1)


def test_nmf_retrouve_deux_themes():
    modele = tp.ajuster_nmf(BOW.astype(float), k=2)
    mots = [set(t) for t in tp.mots_principaux(modele.components_, TERMES, n=3)]
    assert {"pain", "four", "sel"} in mots and {"fusee", "orbite", "lune"} in mots


def test_stabilite_retrouve_une_permutation():
    a = [["a", "b"], ["c", "d"], ["e", "f"]]
    b = [["e", "f"], ["a", "b"], ["c", "d"]]
    resultat = tp.stabilite(a, b)
    assert resultat["jaccard_moyen"] == 1.0
    assert resultat["appariement"] == [1, 2, 0]


# --- Comparaison par polarité -------------------------------------------------------------------

def test_log_odds_conforme_aux_equations_de_monroe():
    """Valeurs calculées à la main avec les équations 16, 19, 22 et 23."""
    sortie = tp.log_odds_dirichlet(np.array([10, 90]), np.array([30, 70]), alpha0=10)
    # a_w = 10 * (40, 160) / 200 = (2, 8) ; n_i = n_j = 100
    delta = math.log(12 / 98) - math.log(32 / 78)
    variance = 1 / 12 + 1 / 98 + 1 / 32 + 1 / 78
    assert sortie.loc[0, "delta"] == pytest.approx(delta)
    assert sortie.loc[0, "variance"] == pytest.approx(variance)
    assert sortie.loc[0, "z"] == pytest.approx(delta / math.sqrt(variance))
    assert sortie.loc[0, "z"] < 0 < sortie.loc[1, "z"]


def test_log_odds_antisymetrique_et_nul_si_groupes_identiques():
    i, j = np.array([12, 50, 7]), np.array([30, 20, 9])
    assert np.allclose(tp.log_odds_dirichlet(i, j)["z"], -tp.log_odds_dirichlet(j, i)["z"])
    assert np.allclose(tp.log_odds_dirichlet(i, i)["z"], 0)


def test_log_odds_refuse_les_entrees_invalides():
    with pytest.raises(ValueError):
        tp.log_odds_dirichlet(np.array([1, 2]), np.array([1, 2]), alpha0=0)
    with pytest.raises(ValueError):
        tp.log_odds_dirichlet(np.array([0, 2]), np.array([0, 2]))
    with pytest.raises(ValueError):
        tp.log_odds_dirichlet(np.array([1]), np.array([1, 2]))


def test_probabilite_superiorite():
    assert tp.probabilite_superiorite(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert tp.probabilite_superiorite(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5


def test_comparer_prevalences_mesure_un_ecart_connu():
    rng = np.random.default_rng(0)
    n = 4000
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    part = np.clip(rng.normal(0.5 + 0.1 * (y - 0.5) * 2, 0.1), 0, 1)
    theta = np.column_stack([part, 1 - part])
    table = tp.comparer_prevalences(theta, y, n_bootstrap=300)
    ligne = table.iloc[0]
    assert ligne["ecart_points"] == pytest.approx(20, abs=1.5)
    assert ligne["ic95_bas"] < ligne["ecart_points"] < ligne["ic95_haut"]
    assert ligne["d_cohen"] == pytest.approx(2.0, abs=0.2)
    assert table.iloc[1]["ecart_points"] == pytest.approx(-ligne["ecart_points"])


def test_comparer_prevalences_reproductible_et_exige_deux_classes():
    rng = np.random.default_rng(1)
    theta = tp.normaliser_lignes(rng.random((200, 3)))
    y = np.array([0, 1] * 100)
    pd.testing.assert_frame_equal(
        tp.comparer_prevalences(theta, y, n_bootstrap=50),
        tp.comparer_prevalences(theta, y, n_bootstrap=50),
    )
    with pytest.raises(ValueError):
        tp.comparer_prevalences(theta, np.zeros(200, dtype=int), n_bootstrap=10)


# --- Extraits -----------------------------------------------------------------------------------

def test_extrait_contient_les_mots_du_topic_et_respecte_la_taille():
    texte = " ".join(["vide"] * 100 + ["zombie", "horror", "zombies"] + ["vide"] * 100)
    extrait = tp.extrait_autour_des_mots(texte, ["zombie", "horror"], taille=60)
    assert "zombie" in extrait and "horror" in extrait
    assert len(extrait) <= 60 + 4  # points de suspension compris
    assert extrait.startswith("…") and extrait.endswith("…")


def test_documents_representatifs_ecarte_courts_et_quasi_copies():
    parts = np.array([0.9, 0.8, 0.7, 0.6])
    longueurs = np.array([100, 100, 10, 100])
    textes = ["a b c d", "a b c d", "e f g h", "i j k l"]
    assert tp.documents_representatifs(parts, longueurs, textes, n=2) == [0, 3]


def test_documents_representatifs_ecarte_le_texte_repete_et_departage():
    parts = np.array([1.0, 1.0, 1.0])
    longueurs = np.array([100, 100, 100])
    textes = [" ".join(["oz great show"] * 20), "a b c d", "e f g h"]
    assert tp.documents_representatifs(parts, longueurs, textes, n=3) == [1, 2]
    departage = np.array([0.0, 0.1, 0.9])
    assert tp.documents_representatifs(parts, longueurs, textes, n=2, departage=departage) == [2, 1]


# --- Garde-fous anti-fuite -----------------------------------------------------------------------

@pytest.mark.parametrize("fichier", ["src/features.py", "src/preprocessing.py", "src/data.py"])
def test_le_socle_supervise_n_importe_pas_les_topics(fichier):
    arbre = ast.parse(Path(RACINE / fichier).read_text(encoding="utf-8"))
    modules = set()
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Import):
            modules.update(a.name for a in noeud.names)
        elif isinstance(noeud, ast.ImportFrom):
            modules.add(noeud.module or "")
            modules.update(f"{noeud.module}.{a.name}" for a in noeud.names)
    assert not any("topics" in m for m in modules)


@pytest.mark.skipif(not CACHES_PRESENTS, reason="caches de la phase 3 absents")
def test_matrices_reelles_alignees_et_sans_etiquette():
    donnees = tp.charger_matrices_topics()
    assert set(donnees) == {"bow", "tfidf", "termes", "id_brut"}
    assert donnees["bow"].shape == donnees["tfidf"].shape == (49575, 21172)
    assert len(donnees["termes"]) == 21172
    y = tp.charger_etiquettes_descriptives(donnees["id_brut"][:100])
    assert set(np.unique(y)) <= {0, 1} and len(y) == 100

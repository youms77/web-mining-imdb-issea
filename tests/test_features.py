"""Tests de la vectorisation : fabriques, absence de fuite, séparation entraînement/test.

Lancement : .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_features.py
Les tests sur le vrai corpus sont ignorés si les caches de data/interim sont absents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src import features as f
from src import features_eval as fe

TEXTES = [
    "not good at all", "very bad film", "bad acting not good", "awful and bad",
    "very good film", "great acting", "good story great film", "wonderful and great",
] * 3
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 3)
CACHES_PRESENTS = f.CHEMIN_TEXTES.exists() and f.CHEMIN_INDEX.exists()


# --- Fabriques ---------------------------------------------------------------------------------

def test_fabriques_renvoient_le_bon_type():
    assert isinstance(f.creer_bow(), CountVectorizer)
    assert not isinstance(f.creer_bow(), TfidfVectorizer)
    assert isinstance(f.creer_tfidf(), TfidfVectorizer)


@pytest.mark.parametrize("nature", ["bow", "tfidf"])
def test_reglages_transmis(nature):
    v = f.creer_vectoriseur(nature, ngram_range=(1, 3), min_df=2, max_df=0.8, max_features=10)
    assert (v.ngram_range, v.min_df, v.max_df, v.max_features) == ((1, 3), 2, 0.8, 10)
    assert v.lowercase is False


def test_sublinear_transmis_pour_tfidf():
    assert f.creer_tfidf(sublinear_tf=True).sublinear_tf is True


def test_sublinear_refuse_pour_bow():
    with pytest.raises(ValueError):
        f.creer_vectoriseur("bow", sublinear_tf=True)


def test_nature_inconnue_refusee():
    with pytest.raises(ValueError):
        f.creer_vectoriseur("hachage")


def test_jetons_du_pretraitement_gardes():
    """Les notes et les expressions forment un seul jeton."""
    v = f.creer_bow().fit(["8/10 special_effect not"])
    assert set(v.vocabulary_) == {"8/10", "special_effect", "not"}


def test_bow_compte_et_tfidf_normalise():
    x_bow = f.creer_bow().fit_transform(["bad bad good"]).toarray()
    assert sorted(x_bow.ravel().tolist()) == [1, 2]
    x_tfidf = f.creer_tfidf().fit_transform(["bad bad good", "good film"])
    assert np.allclose(sparse.linalg.norm(x_tfidf, axis=1), 1.0)


def test_sublinear_attenue_la_repetition():
    textes = ["bad " * 50 + "good", "good film"]
    brut = f.creer_tfidf().fit_transform(textes).toarray()[0].max()
    attenue = f.creer_tfidf(sublinear_tf=True).fit_transform(textes).toarray()[0].max()
    assert attenue < brut


def test_min_df_et_max_df_filtrent():
    textes = ["commun rare", "commun moyen", "commun moyen"]
    assert set(f.creer_bow(min_df=2).fit(textes).vocabulary_) == {"commun", "moyen"}
    assert set(f.creer_bow(max_df=0.7).fit(textes).vocabulary_) == {"rare", "moyen"}


def test_masque_identique_a_scikit_learn():
    """Le tableau de vocabulaire applique exactement la règle du vectoriseur."""
    textes = [" ".join(f"m{j}" for j in range(i % 7 + 1)) for i in range(60)]
    matrice = f.creer_bow().fit_transform(textes).tocsc()
    df = np.diff(matrice.indptr)
    for min_df, max_df in [(1, 1.0), (5, 0.9), (10, 0.5), (2, 0.3)]:
        attendu = len(f.creer_bow(min_df=min_df, max_df=max_df).fit(textes).vocabulary_)
        assert int(fe._masque_termes(df, len(textes), min_df, max_df).sum()) == attendu


def test_profil_matrice():
    profil = f.profiler_matrice(sparse.csr_matrix([[1, 0, 0, 0], [0, 2, 0, 0]]))
    assert profil["documents"] == 2 and profil["termes"] == 4
    assert profil["valeurs_non_nulles"] == 2 and profil["taux_de_creux_pct"] == 75.0


@pytest.mark.parametrize(
    "modele,attendu", [("naive_bayes", CountVectorizer), ("regression_logistique", TfidfVectorizer), ("svm_lineaire", TfidfVectorizer)]
)
def test_association_du_cahier(modele, attendu):
    pipeline = f.creer_pipeline_supervise(modele)
    assert isinstance(pipeline, Pipeline)
    assert type(pipeline.named_steps["vectoriseur"]) is attendu


def test_vectoriseurs_topics():
    assert type(f.creer_vectoriseur_topics("lda")) is CountVectorizer
    assert type(f.creer_vectoriseur_topics("nmf")) is TfidfVectorizer
    with pytest.raises(ValueError):
        f.creer_vectoriseur_topics("bertopic")


def test_modele_inconnu_refuse():
    with pytest.raises(ValueError):
        f.creer_pipeline_supervise("foret")


# --- Absence de fuite --------------------------------------------------------------------------

def test_pipeline_ignore_les_mots_du_test():
    pipeline = f.creer_pipeline_supervise("regression_logistique", min_df=1).fit(TEXTES, Y)
    assert "jetonfuite" not in pipeline.named_steps["vectoriseur"].vocabulary_
    pipeline.predict(["jetonfuite jetonfuite"])  # transformer ne doit rien apprendre
    assert "jetonfuite" not in pipeline.named_steps["vectoriseur"].vocabulary_


def test_vocabulaire_et_idf_independants_du_test():
    """Changer entièrement le test ne change ni le vocabulaire ni les poids IDF."""
    tableau = pd.DataFrame({
        "texte_a": TEXTES + ["film test original"] * 4,
        "partition": ["entrainement"] * len(TEXTES) + ["test"] * 4,
    })
    modifie = tableau.copy()
    modifie.loc[modifie["partition"] == "test", "texte_a"] = "jetonfuite good good bad"
    a = f.ajuster_sur_entrainement(f.creer_tfidf(), tableau)
    b = f.ajuster_sur_entrainement(f.creer_tfidf(), modifie)
    assert a.vocabulary_ == b.vocabulary_
    assert np.array_equal(a.idf_, b.idf_)
    assert "jetonfuite" not in b.vocabulary_ and "original" not in a.vocabulary_


def test_ajustement_exige_la_partition():
    with pytest.raises(ValueError):
        f.ajuster_sur_entrainement(f.creer_bow(), pd.DataFrame({"texte_a": ["a b"]}))
    with pytest.raises(ValueError):
        f.ajuster_sur_entrainement(f.creer_bow(), pd.DataFrame({"texte_a": ["a b"], "partition": ["test"]}))


def test_validation_croisee_reajuste_dans_chaque_pli():
    """Un mot présent uniquement dans le pli de validation reste hors du vocabulaire du pli."""
    textes = np.array([f"{t} unique{i}" for i, t in enumerate(TEXTES)], dtype=object)
    plis = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(textes, Y))
    resultat = cross_validate(
        f.creer_pipeline_supervise("regression_logistique", ngram_range=(1, 1), min_df=1),
        textes, Y, cv=plis, return_estimator=True,
    )
    for modele, (_, validation) in zip(resultat["estimator"], plis):
        vocabulaire = modele.named_steps["vectoriseur"].vocabulary_
        assert not {f"unique{i}" for i in validation} & set(vocabulaire)


def test_pipeline_supervise_independant_du_pipeline_b(monkeypatch):
    """Les listes du pipeline B, calculées sur tout le corpus, ne servent à aucun modèle supervisé."""
    from src import preprocessing as pp

    def interdit(*_args, **_kwargs):
        raise AssertionError("Les ressources du pipeline B ont été lues.")

    monkeypatch.setattr(pp, "charger_ressources_b", interdit)
    textes = [pp.pipeline_a(t) for t in TEXTES]
    modele = f.creer_pipeline_supervise("regression_logistique", ngram_range=(1, 1), min_df=1).fit(textes, Y)
    assert "not" in modele.named_steps["vectoriseur"].vocabulary_


def test_decision_appariee():
    reference = {"exactitude_par_pli": [0.90, 0.91, 0.90, 0.91, 0.90]}
    meilleur = {"exactitude_par_pli": [0.91, 0.92, 0.91, 0.92, 0.911]}
    bruit = {"exactitude_par_pli": [0.901, 0.909, 0.902, 0.908, 0.901]}
    assert fe.decider(reference, meilleur)["ameliore"] is True
    assert fe.decider(meilleur, reference)["degrade"] is True
    assert fe.decider(reference, bruit)["ameliore"] is False


# --- Index et quasi-doublons -------------------------------------------------------------------

def test_index_chevauchants_refuses(tmp_path):
    with pytest.raises(ValueError):
        f.sauvegarder_index([1, 2, 3], [3, 4], tmp_path / "index.parquet")


def test_index_aller_retour(tmp_path):
    chemin = f.sauvegarder_index([5, 1, 3], [2, 4], tmp_path / "index.parquet")
    train, test = f.charger_index(chemin)
    assert sorted(train.tolist()) == [1, 3, 5] and sorted(test.tolist()) == [2, 4]


def test_empreinte_independante_de_l_ordre():
    assert f.empreinte_index([3, 1, 2]) == f.empreinte_index([1, 2, 3])
    assert f.empreinte_index([1, 2, 3]) != f.empreinte_index([1, 2, 4])


def test_quasi_doublon_detecte():
    base = " ".join(f"mot{i}" for i in range(40))
    modifie = base.replace("mot20", "autre")
    paires = fe.mesurer_quasi_doublons([base, "rien de commun ici"], [modifie], taille=5)
    ligne = paires.sort_values("contenance").iloc[-1]
    assert ligne["i_train"] == 0 and ligne["j_test"] == 0
    assert 0.8 < ligne["contenance"] < 1


# --- Vrai corpus -------------------------------------------------------------------------------

@pytest.mark.skipif(not CACHES_PRESENTS, reason="caches de data/interim absents")
def test_vrais_index_disjoints_et_complets():
    train, test = f.charger_index()
    assert len(train) == 39660 and len(test) == 9915
    assert np.intersect1d(train, test).size == 0
    cache = pd.read_parquet(f.CHEMIN_TEXTES, columns=["id_brut", "partition"])
    assert set(cache.loc[cache["partition"] == "entrainement", "id_brut"]) == set(train.tolist())
    assert set(cache.loc[cache["partition"] == "test", "id_brut"]) == set(test.tolist())


@pytest.mark.skipif(not CACHES_PRESENTS, reason="caches de data/interim absents")
def test_vrais_textes_sans_texte_commun():
    """Le texte d'entrée des modèles supervisés (pipeline A) n'est jamais commun aux deux parties."""
    d = pd.read_parquet(f.CHEMIN_TEXTES, columns=["partition", "texte_a"])
    masque = d["partition"] == "entrainement"
    assert not set(d.loc[masque, "texte_a"]) & set(d.loc[~masque, "texte_a"])


@pytest.mark.skipif(not CACHES_PRESENTS, reason="caches de data/interim absents")
def test_chargement_entrainement_sans_ligne_de_test():
    train_ids, test_ids = f.charger_index()
    lu = f.charger_entrainement()
    assert set(lu["id_brut"]) == set(train_ids.tolist())
    assert not set(lu["id_brut"]) & set(test_ids.tolist())


@pytest.mark.skipif(not CACHES_PRESENTS, reason="caches de data/interim absents")
def test_vrai_vocabulaire_independant_du_test():
    """Remplacer les vrais textes de test ne modifie ni le vocabulaire ni les IDF."""
    d = pd.read_parquet(f.CHEMIN_TEXTES, columns=["partition", "texte_a"])
    modifie = d.copy()
    modifie.loc[modifie["partition"] == "test", "texte_a"] = "jetonfuite jetonfuite"
    a = f.ajuster_sur_entrainement(f.creer_tfidf(min_df=5), d)
    b = f.ajuster_sur_entrainement(f.creer_tfidf(min_df=5), modifie)
    assert a.vocabulary_ == b.vocabulary_
    assert np.array_equal(a.idf_, b.idf_)
    assert "jetonfuite" not in b.vocabulary_

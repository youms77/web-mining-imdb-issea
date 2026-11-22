"""Tests de la phase 5 : équivalence du pipeline accéléré, calibration, métriques, zone."""
import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

from src import evaluation as ev
from src import features as f
from src import models as m
from src.exploration import _NOTE
from src.preprocessing import MOTIF_NOTE, retirer_notes

POSITIFS = ["great film loved it", "wonderful acting great story", "loved this wonderful film",
            "excellent story great cast", "superb film loved cast", "great wonderful excellent film"]
NEGATIFS = ["bad film hated it", "awful acting bad story", "hated this awful film",
            "terrible story bad cast", "boring film hated cast", "bad awful terrible film"]
TEXTES = np.array((POSITIFS + NEGATIFS) * 5, dtype=object)
Y = np.array(([1] * 6 + [0] * 6) * 5)


@pytest.mark.parametrize("famille", list(m.FAMILLES))
@pytest.mark.parametrize("min_df,max_df", [(5, 1.0), (10, 0.5), (5, 0.3)])
def test_pipeline_comptages_egal_pipeline_texte(famille, min_df, max_df):
    """Filtrer des comptages (min_df=1 ici) donne exactement le modèle appris sur le texte."""
    hp = {m.GRILLE_CLASSIFIEUR[m.FAMILLES[famille][0]][0]: m.GRILLE_CLASSIFIEUR[m.FAMILLES[famille][0]][1][1]}
    try:
        texte = m.creer_pipeline_texte(famille, hp, min_df=min_df, max_df=max_df).fit(TEXTES, Y)
    except ValueError:
        pytest.skip("Vocabulaire vide pour ce réglage sur le mini-corpus.")
    comptages = f.creer_bow(ngram_range=m.NGRAM, min_df=1)
    x = comptages.fit_transform(TEXTES)
    rapide = m.creer_pipeline_comptages(famille, hp, min_df=min_df, max_df=max_df).fit(x, Y)
    assert rapide.named_steps["filtre"].indices_.size == len(texte.named_steps["vectoriseur"].vocabulary_)
    np.testing.assert_allclose(
        m.score_continu(rapide, x), m.score_continu(texte, TEXTES), rtol=1e-6, atol=1e-6
    )


def test_filtre_termes_regle_scikit_learn():
    x = f.creer_bow(min_df=1).fit_transform(TEXTES)
    filtre = m.FiltreTermes(min_df=10, max_df=0.5).fit(x)
    attendu = f.creer_bow(min_df=10, max_df=0.5).fit(TEXTES)
    assert filtre.transform(x).shape[1] == len(attendu.vocabulary_)


def test_filtre_apprend_sur_fit_seulement():
    x = f.creer_bow(min_df=1).fit_transform(TEXTES)
    filtre = m.FiltreTermes(min_df=5).fit(x[:12])
    assert filtre.transform(x).shape[1] == 0 or filtre.indices_.max() < x.shape[1]
    assert filtre.transform(x[12:]).shape[0] == len(TEXTES) - 12


TAILLES_VUES = []


class VectoriseurEspion(CountVectorizer):
    """Note le nombre de textes reçus à chaque apprentissage du vocabulaire."""

    def fit_transform(self, raw_documents, y=None):
        TAILLES_VUES.append(len(raw_documents))
        return super().fit_transform(raw_documents, y)


def test_calibration_reapprend_le_vocabulaire_dans_chaque_pli():
    """Le vocabulaire est appris 5 fois sur 4/5 des textes, puis une fois sur tous les textes."""
    TAILLES_VUES.clear()
    pipeline = Pipeline([("vectoriseur", VectoriseurEspion()), ("classifieur", LogisticRegression())])
    modele = m.calibrer(pipeline, "sigmoid").fit(TEXTES, Y)
    assert sorted(TAILLES_VUES) == [48] * 5 + [60]
    assert modele.predict_proba(["great film"]).shape == (1, 2)
    assert len(modele.calibrated_classifiers_) == 1  # ensemble=False : un seul Pipeline final


def test_calibrer_refuse_methode_inconnue():
    with pytest.raises(ValueError):
        m.calibrer(m.creer_pipeline_texte("svm_lineaire"), "platt")


def test_motif_note_identique_a_la_phase_1():
    assert MOTIF_NOTE == _NOTE.pattern


@pytest.mark.parametrize("avant,apres", [
    ("I give it 8/10 overall", "I give it   overall"),
    ("rated 3 out of 10.", "rated  ."),
    ("7.5/10 for me", "  for me"),
    ("in 24/7 and 1/2", "in 24/7 and 1/2"),
    ("100/10", "100/10"),
])
def test_retirer_notes(avant, apres):
    assert retirer_notes(avant) == apres


def test_preparer_texte_retire_la_note_avant_le_pipeline_a():
    assert m.preparer_texte("I didn't like it, 3/10.<br />Bad!") == "did not like it bad"
    assert m.preparer_texte(None) == ""


def test_metriques_classement_a_la_main():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    r = ev.metriques_classement(y, p)
    assert r["exactitude"] == 0.5 and r["precision"] == 0.5 and r["rappel"] == 0.5
    assert r["matrice_confusion"] == {"vrais_negatifs": 1, "faux_positifs": 1, "faux_negatifs": 1, "vrais_positifs": 1}
    assert r["roc_auc"] == 0.75


def test_brier_et_ece():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.2, 0.6, 0.4])
    assert ev.metriques_probabilite(y, p)["brier"] == pytest.approx(brier_score_loss(y, p))
    # Tranches : 0,2 (fréq. 0), 0,4 (0), 0,6 (1), 0,8 (1) : écart moyen = (0,2+0,4+0,4+0,2)/4.
    assert ev.erreur_calibration(y, p) == pytest.approx(0.3)


def test_bootstrap_encadre_la_valeur():
    tirage = np.random.default_rng(0)
    y = tirage.integers(0, 2, 2000)
    p = np.clip(y * 0.6 + tirage.normal(0.2, 0.2, 2000), 0, 1)
    ic = ev.bootstrap_ic(y, p, n=200)
    valeur = ev.metriques_classement(y, p)["exactitude"]
    assert ic["exactitude"]["bas"] < valeur < ic["exactitude"]["haut"]


def test_zone_atteint_la_cible_et_est_minimale():
    tirage = np.random.default_rng(0)
    p = tirage.uniform(0, 1, 20000)
    y = (tirage.uniform(0, 1, 20000) < p).astype(int)  # probabilités parfaitement calibrées
    zone = ev.choisir_zone_incertitude(y, p, cible=0.8)
    assert zone["exactitude_hors_zone"] >= 0.8
    plus_petite = zone["demi_largeur"] - 0.005
    hors = (p <= 0.5 - plus_petite) | (p >= 0.5 + plus_petite)
    assert ((p[hors] >= 0.5).astype(int) == y[hors]).mean() < 0.8


def test_appliquer_zone():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.55, 0.52])
    r = ev.appliquer_zone(y, p, 0.4, 0.6)
    assert r["effectif_dans_la_zone"] == 2 and r["exactitude_hors_zone"] == 1.0 and r["exactitude_dans_la_zone"] == 0.5


def test_sauvegarde_rechargement_predictions_identiques(tmp_path):
    modele = m.calibrer(m.creer_pipeline_texte("regression_logistique", {"C": 1.0}, min_df=1), "isotonic").fit(TEXTES, Y)
    info = m.sauvegarder_modele(modele, tmp_path / "modele.joblib")
    relu = m.charger_modele(tmp_path / "modele.joblib")
    assert info["taille_mo"] >= 0 and len(info["sha256"]) == 64
    np.testing.assert_array_equal(modele.predict_proba(TEXTES), relu.predict_proba(TEXTES))


def test_vader_seuil_et_score():
    vader = m.ReferenceVader()
    scores = vader.scores(["This movie is wonderful and I love it!", "This movie is horrible and I hate it."])
    assert scores[0] > 0 > scores[1]
    assert vader.predire(scores).tolist() == [1, 0]

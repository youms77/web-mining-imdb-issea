"""Tests de l'application : logique, parcours AppTest, légèreté et fidélité des chiffres."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import application as ap

RACINE = Path(__file__).resolve().parents[1]
APP = RACINE / "app" / "app.py"
INTERDITS = {"torch", "sentence_transformers", "bertopic", "umap", "hdbscan", "transformers", "gensim"}
TEXTE_FRANCAIS = "Ce film est vraiment magnifique, les acteurs sont excellents et l'histoire est très touchante du début à la fin."
TEXTE_COURT = "Good actors, but the story goes nowhere."


@pytest.fixture(scope="module")
def artefacts():
    return ap.charger_artefacts()


@pytest.fixture(scope="module")
def metriques():
    return json.loads((RACINE / "reports" / "metrics.json").read_text(encoding="utf-8"))


def lancer(page: str | None = None) -> AppTest:
    test = AppTest.from_file(str(APP), default_timeout=120).run()
    if page:
        test.switch_page(page).run()
    assert not test.exception
    return test


def fichiers_app() -> list[Path]:
    return sorted((RACINE / "app").rglob("*.py")) + [RACINE / "src" / "application.py"]


def modules_importes(chemin: Path) -> set[str]:
    modules: set[str] = set()
    for noeud in ast.walk(ast.parse(chemin.read_text(encoding="utf-8"))):
        if isinstance(noeud, ast.Import):
            modules |= {a.name.split(".")[0] for a in noeud.names}
        elif isinstance(noeud, ast.ImportFrom) and noeud.module:
            modules.add(noeud.module.split(".")[0])
    return modules


# --- Parcours AppTest exigés ---------------------------------------------------------------------

def test_saisie_vide_message_francais():
    test = lancer()
    test.button(key="analyser").click().run()
    assert not test.exception
    assert "Collez ou écrivez" in test.error[0].value


@pytest.mark.parametrize("cle, categorie", [("positif", "positif"), ("negatif", "negatif")])
def test_exemples_positif_et_negatif(cle, categorie):
    test = lancer()
    test.button(key=f"exemple_{cle}").click().run()
    assert not test.exception
    resultat = test.session_state["resultat"]
    assert resultat["fiable"] and resultat["categorie"] == categorie
    assert test.metric[0].label.startswith("Probabilité calibrée")
    # NMF est creux : un thème de poids nul n'est pas affiché, il peut donc y en avoir moins de 3.
    assert 1 <= len(resultat["themes"]) <= 3 and all(t["part_pct"] > 0 for t in resultat["themes"])


def test_texte_non_anglais_refuse():
    test = lancer()
    test.text_area(key="texte_critique").set_value(TEXTE_FRANCAIS)
    test.button(key="analyser").click().run()
    assert not test.exception
    assert "ne semble pas écrit en anglais" in test.error[0].value
    assert test.session_state["resultat"]["code"] == "non_anglais"


def test_texte_court_score_non_presente_comme_fiable():
    test = lancer()
    test.text_area(key="texte_critique").set_value(TEXTE_COURT)
    test.button(key="analyser").click().run()
    assert not test.exception
    resultat = test.session_state["resultat"]
    assert resultat["code"] == "court" and not resultat["fiable"]
    assert any("Score non fiable" in m.value for m in test.markdown)
    assert any("pas fiable" in w.value for w in test.warning)
    assert not [m for m in test.metric if m.label.startswith("Probabilité calibrée")]


def test_texte_trop_court_et_trop_long(artefacts):
    assert ap.analyser("Great movie!", artefacts)["code"] == "trop_court"
    limite = artefacts.metriques["application"]["seuils_texte"]["caracteres_max"]
    assert ap.analyser("word " * (limite // 5 + 10), artefacts)["code"] == "trop_long"


def test_exemple_mitige_affiche_le_resultat_reel(metriques):
    test = lancer()
    test.button(key="exemple_mitige").click().run()
    assert not test.exception
    info = metriques["application"]["exemple_mitige"]
    assert test.session_state["resultat"]["probabilite_positive"] == pytest.approx(info["probabilite_modele_gele"])
    message = test.info[0].value
    assert ap.fr_nombre(info["probabilite_modele_gele"], 3) in message
    assert ap.fr_nombre(info["probabilite_hors_pli"], 3) in message


def test_echec_de_chargement_message_clair(monkeypatch):
    def echec(*_args, **_kwargs):
        raise ap.ErreurChargement("Un modèle n'a pas pu être chargé depuis deploy.")

    st.cache_resource.clear()
    monkeypatch.setattr(ap, "charger_artefacts", echec)
    try:
        test = AppTest.from_file(str(APP), default_timeout=120).run()
        assert not test.exception
        assert "Échec de chargement" in test.error[0].value
    finally:
        monkeypatch.undo()
        st.cache_resource.clear()


def test_fichier_absent_donne_erreur_de_chargement(tmp_path):
    with pytest.raises(ap.ErreurChargement):
        ap.charger_artefacts(dossier=tmp_path)


@pytest.mark.parametrize("page", ["vues/explorer.py", "vues/performances.py", "vues/methode.py"])
def test_pages_sans_exception(page):
    lancer(page)


@pytest.mark.parametrize("methode", ["LDA", "BERTopic"])
def test_choix_de_methode(methode):
    test = lancer("vues/explorer.py")
    test.segmented_control(key="methode_themes").set_value(methode).run()
    assert not test.exception


# --- Fidélité des chiffres -----------------------------------------------------------------------

def test_seuils_lus_dans_metrics_pas_en_dur():
    for chemin in fichiers_app():
        contenu = chemin.read_text(encoding="utf-8")
        assert not re.search(r"(?<![\d.])0[.,](?:28|72)(?!\d)", contenu), chemin.name


def test_regle_de_zone_identique_a_la_phase_5(metriques):
    zone = metriques["sentiment"]["choix_figes"]["zone_incertitude"]
    bas, haut = zone["seuil_bas"], zone["seuil_haut"]
    assert ap.categoriser(bas, bas, haut) == "negatif"
    assert ap.categoriser(haut, bas, haut) == "positif"
    assert ap.categoriser((bas + haut) / 2, bas, haut) == "incertain"


def test_performances_identiques_a_metrics(metriques):
    test = lancer("vues/performances.py")
    resume = metriques["sentiment"]["resume_modele_retenu"]
    valeurs = {m.label: m.value for m in test.metric}
    assert valeurs["Exactitude"] == ap.fr_pct(resume["exactitude"], 2)
    assert valeurs["Score F1"] == ap.fr_nombre(resume["f1"], 3)
    assert valeurs["ROC-AUC"] == ap.fr_nombre(resume["roc_auc"], 3)
    zone = metriques["sentiment"]["test"]["zone_incertitude"]
    assert valeurs["Dans la zone"] == ap.fr_pct(zone["exactitude_dans_la_zone"])
    assert valeurs["Hors de la zone"] == ap.fr_pct(zone["exactitude_hors_zone"])


def test_bertopic_presente_honnetement(metriques):
    test = lancer("vues/explorer.py")
    test.segmented_control(key="methode_themes").set_value("BERTopic").run()
    alerte = test.warning[0].value
    assert ap.fr_entier(metriques["bertopic"]["echantillon"]["critiques"]) in alerte
    assert ap.fr_pct(metriques["bertopic"]["concentration"]["bertopic_k8"]["part_hors_topic"]) in alerte
    assert "NMF reste le modèle commun" in alerte
    assert any("exploratoire" in e.label.lower() for e in test.expander)


def test_cles_existantes_de_metrics_inchangees(metriques):
    # Version publiée : le dépôt n'a pas l'historique du projet, donc le commit de référence est absent.
    try:
        sortie = subprocess.run(
            ["git", "show", "967a7e7:reports/metrics.json"], cwd=RACINE, capture_output=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("commit de référence absent : test réservé au dépôt de travail")
    reference = json.loads(sortie.decode("utf-8"))
    for cle, valeur in reference.items():
        assert metriques[cle] == valeur, cle


# --- Logique -------------------------------------------------------------------------------------

def test_contributions_somment_a_la_decision(artefacts):
    texte = artefacts.contenu["exemples"]["mitige"]["texte"]
    jetons = ap.preparer_texte(texte).split()
    contributions = ap.contributions_par_position(jetons, artefacts.vectoriseur, artefacts.coefficients)
    base = artefacts.modele.calibrated_classifiers_[0].estimator
    decision = float(base.decision_function([" ".join(jetons)])[0])
    assert contributions.sum() + artefacts.constante == pytest.approx(decision, abs=1e-9)


def test_texte_saisi_echappe(artefacts):
    texte = artefacts.contenu["exemples"]["negatif"]["texte"] + " <script>alert('x')</script> <img src=x onerror=alert(1)>"
    sortie = ap.analyser(texte, artefacts)["html_surligne"]
    assert "<script" not in sortie and "<img" not in sortie
    assert "&lt;script&gt;" in sortie


def test_surlignage_conserve_le_texte(artefacts):
    texte = artefacts.contenu["exemples"]["mitige"]["texte"]
    jetons = ap.preparer_texte(texte).split()
    contributions = ap.contributions_par_position(jetons, artefacts.vectoriseur, artefacts.coefficients)
    segments = ap.segments_surlignes(texte, jetons, ap.mots_influents(jetons, contributions))
    assert "".join(s for s, _ in segments) == texte
    assert any(n == "pos" for _, n in segments) and any(n == "neg" for _, n in segments)


def test_parts_de_themes_sont_des_poids(artefacts):
    themes = ap.themes_principaux(artefacts.contenu["exemples"]["positif"]["texte"], artefacts, n=8)
    assert sum(t["part_pct"] for t in themes) == pytest.approx(100, abs=1e-6)
    assert [t["part_pct"] for t in themes] == sorted((t["part_pct"] for t in themes), reverse=True)


def test_interface_precise_que_les_parts_ne_sont_pas_des_probabilites():
    test = lancer()
    test.button(key="exemple_positif").click().run()
    assert any("pas des probabilités" in c.value for c in test.caption)


@pytest.mark.parametrize("texte", [
    "Dieser Film ist wirklich wunderbar, die Schauspieler sind ausgezeichnet und die Geschichte ist sehr bewegend.",
    "Esta película es maravillosa, los actores son excelentes y la historia es muy conmovedora de principio a fin.",
    "これは本当に素晴らしい映画でした。俳優も物語も最高です。",
])
def test_autres_langues_refusees(texte, artefacts):
    assert ap.analyser(texte, artefacts)["code"] == "non_anglais"


def test_anglais_court_sans_mot_vide_accepte(artefacts):
    assert ap.analyser("Brilliant acting, superb direction, wonderful score throughout.", artefacts)["code"] == "court"


# --- Légèreté et déploiement ---------------------------------------------------------------------

@pytest.mark.parametrize("chemin", fichiers_app(), ids=lambda c: c.name)
def test_aucun_import_lourd_dans_app(chemin):
    assert not modules_importes(chemin) & INTERDITS


def test_import_reel_sans_paquet_lourd():
    code = (
        "import sys; sys.path[:0] = ['.', 'app']; import interface; "
        f"print(sorted(m for m in {sorted(INTERDITS)!r} if m in sys.modules))"
    )
    sortie = subprocess.run([sys.executable, "-c", code], cwd=RACINE, capture_output=True, text=True, check=True)
    assert sortie.stdout.strip().splitlines()[-1] == "[]"


def test_requirements_app_legers_et_epingles():
    lignes = [
        l.strip() for l in (RACINE / "app" / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    ]
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]+==[0-9][0-9a-zA-Z.]*", l) for l in lignes)
    noms = {l.split("==")[0].lower() for l in lignes}
    assert not noms & {"torch", "bertopic", "transformers", "sentence-transformers", "umap-learn", "hdbscan", "gensim"}
    socle = dict(l.split("==") for l in (RACINE / "requirements.txt").read_text(encoding="utf-8").splitlines() if "==" in l)
    versions = dict(l.split("==") for l in lignes)
    assert versions["scikit-learn"] == socle["scikit-learn"]


def test_modele_deploye_identique_au_modele_gele(metriques):
    attendu = metriques["sentiment"]["choix_figes"]["modeles"]["svm_lineaire"]["sha256"]
    contenu = (ap.DOSSIER_DEPLOIEMENT / ap.FICHIER_SVM).read_bytes()
    assert hashlib.sha256(contenu).hexdigest().upper() == attendu


def test_deploiement_sans_corpus_et_leger():
    fichiers = [f for f in ap.DOSSIER_DEPLOIEMENT.rglob("*") if f.is_file()]
    assert not [f for f in fichiers if f.suffix in {".csv", ".parquet", ".npy", ".npz"}]
    assert sum(f.stat().st_size for f in fichiers) < 25 * 1024 ** 2


def test_lemmatisation_sans_data_interim():
    code = (
        "from pathlib import Path; import src.preprocessing as p; "
        "p.DOSSIER_NLTK = Path('inexistant'); print(p.pipeline_b('The special effects were stunning'))"
    )
    sortie = subprocess.run([sys.executable, "-c", code], cwd=RACINE, capture_output=True, text=True, check=True)
    assert "special_effect" in sortie.stdout


def test_extraits_courts_et_listes(artefacts):
    extraits = artefacts.contenu["extraits"]
    assert len(extraits) <= artefacts.metriques["topics"]["modele_commun"]["k"]
    assert all(len(e["extrait"]) <= 170 for e in extraits)
    liste = (RACINE / "docs" / "extraits_application.md").read_text(encoding="utf-8")
    assert all(str(e["id_brut"]) in liste for e in extraits)
    assert not any(re.search(r"https?://|www\.|@", e["extrait"]) for e in extraits)


def test_probabilites_du_modele_deploye_identiques(artefacts):
    if not (RACINE / "models" / ap.FICHIER_SVM).exists():
        pytest.skip("modèle gelé complet non publié : test réservé au dépôt de travail")
    source = ap.joblib.load(RACINE / "models" / ap.FICHIER_SVM)
    textes = [e["texte"] for e in artefacts.contenu["exemples"].values()]
    preparés = [ap.preparer_texte(t) for t in textes]
    assert np.array_equal(source.predict_proba(preparés), artefacts.modele.predict_proba(preparés))

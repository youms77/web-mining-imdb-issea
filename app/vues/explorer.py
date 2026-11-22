"""Page « Exploration des thèmes » : NMF (par défaut), LDA et BERTopic (résultats calculés sur Colab)."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

import interface as ui
from src import application as ap

artefacts = ui.charger()
metriques = artefacts.metriques
topics = metriques["topics"]
colab = metriques["bertopic_colab"]
extraits = {e["topic"]: e for e in artefacts.contenu.get("extraits", [])}
METHODES = {"NMF (par défaut)": "nmf", "LDA": "lda", "BERTopic (analyse approfondie)": "bertopic"}
NATURES = {"sujet": "Sujet", "jugement": "Jugement", "mixte": "Mixte"}


def afficher_fiche(cle: str) -> None:
    """Fiche d'un thème NMF ou LDA : mots, présence par polarité et extrait éventuel."""
    fiches = {f["topic"]: f for f in topics["fiches_topics"][cle]}
    polarite = {l["topic"]: l for l in topics["comparaison_polarite"][cle]["topics"]}
    ordre = [l["topic"] for l in topics["comparaison_polarite"][cle]["topics"]]
    choix = st.selectbox("Choisir un thème", ordre, format_func=lambda t: fiches[t]["nom"], key=f"fiche_{cle}")
    fiche, ligne = fiches[choix], polarite[choix]
    with ui.carte(f"fiche_{cle}", ton="orange"):
        st.subheader(fiche["nom"], anchor=False)
        st.markdown(f"**Nature :** {NATURES.get(fiche['nature'], fiche['nature'])}")
        st.markdown(f"**Mots principaux :** {', '.join(fiche['mots'])}")
        st.markdown(fiche["lecture"])
        ui.indicateurs([
            ("Avis positifs", f"{ap.fr_nombre(ligne['prevalence_positif_pct'], 1)} %", "vert"),
            ("Avis négatifs", f"{ap.fr_nombre(ligne['prevalence_negatif_pct'], 1)} %", "rouge"),
            ("Écart", ui.fr_ecart(ligne["ecart_points"]), "bleu"),
        ])
        st.markdown(
            f"**IC à 95 % de l'écart :** {ap.fr_nombre(ligne['ic95_bas'], 1)} à {ap.fr_nombre(ligne['ic95_haut'], 1)} points. "
            f"**d de Cohen :** {ap.fr_nombre(ligne['d_cohen'], 2)}."
        )
        extrait = extraits.get(choix) if cle == "nmf" else None
        if extrait and extrait.get("extrait"):
            st.markdown("**Extrait d'une critique** (texte nettoyé)")
            st.html(f'<div class="extrait">{escape(extrait["extrait"])}</div>')


def afficher_cours(cle: str) -> None:
    lignes = topics["comparaison_polarite"][cle]["topics"]
    choix_k = topics["choix_k"][cle]
    ui.indicateurs([
        ("Nombre de thèmes", choix_k["k"], "bleu"),
        ("Cohérence c_v", ap.fr_nombre(choix_k["c_v"], 3), "vert"),
        ("Diversité des mots", ap.fr_nombre(choix_k["diversite"], 3), "orange"),
        ("Critiques modélisées", ap.fr_entier(topics["corpus"]["documents"]), "bleu"),
    ])
    col_graphe, col_fiche = st.columns([1.1, 1], gap="large")
    with col_graphe:
        with ui.carte(f"polarite_{cle}"):
            st.subheader("Prévalence des thèmes selon la polarité", anchor=False)
            ui.afficher(ui.barres_divergentes(lignes), f"polarite_{cle}", "Écart de présence de chaque thème entre avis positifs et négatifs.")
    with col_fiche:
        afficher_fiche(cle)


def afficher_bertopic() -> None:
    reglage = colab["reglage_retenu"]
    ui.indicateurs([
        ("Thèmes (k-means)", reglage["n_themes"], "bleu"),
        ("Critiques encodées", ap.fr_entier(colab["critiques"]), "vert"),
        ("Cohérence c_v", ap.fr_nombre(reglage["c_v"], 3), "orange"),
        ("Stabilité (Jaccard moyen)", ap.fr_nombre(reglage["stabilite_jaccard_moyen"], 3), "rouge"),
    ])
    with ui.carte("reglage_colab", ton="orange"):
        st.subheader("Réglage retenu pour BERTopic", anchor=False)
        grille = colab["grille"]
        st.markdown(
            f"- **Encodeur :** {colab['encodeur']['nom']} (licence {colab['encodeur']['licence']}), sur {ap.fr_entier(colab['critiques'])} critiques.\n"
            f"- **Texte :** sans les mots génériques (movie, film, watch…). **Réduction :** {reglage['reduction']}.\n"
            f"- **Regroupement :** k-means à {reglage['n_themes']} thèmes. {grille['reglages_hdbscan']} réglages HDBSCAN ont été essayés : "
            f"ils donnent au plus {grille['hdbscan_themes_max']} thèmes, dont un contient au moins "
            f"{ap.fr_pct(grille['hdbscan_plus_grand_theme_min'], 0)} du corpus.\n"
            f"- **Axe 1 retiré :** {reglage['raison_axe_1'].replace('0.', '0,')}.\n"
            f"- **Stabilité mesurée :** Jaccard moyen {ap.fr_nombre(reglage['stabilite_jaccard_moyen'], 3)} sur des sous-échantillons de "
            f"{ap.fr_pct(reglage['stabilite_part_sous_echantillon'], 0)}, un thème sur trois environ change de composition d'un tirage à l'autre.\n"
            f"- **Hors thème :** {ap.fr_pct(reglage['part_hors_theme'], 0)} (k-means classe toutes les critiques). Le plus grand thème contient "
            f"{ap.fr_pct(reglage['plus_grand_theme_part'])} du corpus.\n"
            "- **NMF reste le modèle par défaut** de l'application ; BERTopic sert d'analyse approfondie."
        )

    col_carte, col_polarite = st.columns([1, 1.2], gap="large")
    with col_carte:
        with ui.carte("carte_colab"):
            st.subheader("Carte des critiques en deux dimensions", anchor=False)
            noms = {t["topic"]: t["nom"] for t in colab["themes"]}
            ui.afficher(ui.nuage_themes(ui.agregats()["carte_colab"], noms), "carte_colab", "Échantillon de critiques projetées en deux dimensions, colorées selon la polarité.")
    with col_polarite:
        with ui.carte("polarite_colab"):
            st.subheader("Prévalence des thèmes selon la polarité", anchor=False)
            ui.afficher(ui.barres_divergentes(colab["polarite"]["themes"], hauteur_ligne=36), "polarite_colab", "Écart de présence de chaque thème BERTopic entre avis positifs et négatifs.")

    themes = {t["topic"]: t for t in colab["themes"]}
    lignes = {l["topic"]: l for l in colab["polarite"]["themes"]}
    choix = st.selectbox("Choisir un thème", list(themes), format_func=lambda t: f"{t}. {themes[t]['nom']}", key="fiche_colab")
    t, l = themes[choix], lignes[choix]
    with ui.carte("fiche_colab", ton="orange"):
        st.subheader(t["nom"], anchor=False)
        st.markdown(f"**Mots principaux :** {', '.join(t['mots']).replace('_', ' ')}")
        ui.indicateurs([
            ("Critiques du thème", ap.fr_entier(t["effectif"]), "bleu", f"{ap.fr_nombre(t['part_corpus_pct'], 1)} % du corpus"),
            ("Avis positifs", ap.fr_entier(t["avis_positifs"]), "vert", f"{ap.fr_nombre(l['prevalence_positif_pct'], 1)} % des avis positifs"),
            ("Avis négatifs", ap.fr_entier(t["avis_negatifs"]), "rouge", f"{ap.fr_nombre(l['prevalence_negatif_pct'], 1)} % des avis négatifs"),
            ("Écart", ui.fr_ecart(l["ecart_points"]), "orange", f"IC 95 % : {ap.fr_nombre(l['ic95_bas'], 1)} à {ap.fr_nombre(l['ic95_haut'], 1)}"),
        ])

    phrases = metriques["bertopic"]["extension_phrases"]
    with st.expander("Extension exploratoire : thèmes et sentiment par phrase", icon=":material/science:"):
        st.warning(phrases["avertissement"], icon=":material/science:")
        mixtes = phrases["critiques_mixtes"]["seuils_figes"]
        ui.indicateurs([
            ("Phrases scorées", ap.fr_entier(phrases["phrases_scorees"]), "bleu"),
            ("Phrases hors aspect", ap.fr_pct(phrases["part_phrases_hors_aspect"]), "orange"),
            ("Avis positifs à phrase négative", ap.fr_pct(mixtes["critiques_positives_avec_une_phrase_negative"]), "rouge"),
        ])


def afficher_comparaison() -> None:
    """Tableau des trois méthodes, sur leurs propres corpus et réglages."""
    reglage = colab["reglage_retenu"]
    nmf, lda = topics["choix_k"]["nmf"], topics["choix_k"]["lda"]
    jac = {c: topics["stabilite"].get(c, {}).get("jaccard_moyen") for c in ("nmf", "lda")}
    with ui.carte("comparaison_methodes"):
        st.subheader("Comparaison des trois méthodes de modélisation thématique", anchor=False)
        st.table(pd.DataFrame({
            "NMF (par défaut)": {
                "Nombre de thèmes": nmf["k"], "Cohérence c_v": ap.fr_nombre(nmf["c_v"], 3), "Diversité": ap.fr_nombre(nmf["diversite"], 3),
                "Stabilité (Jaccard)": ap.fr_nombre(jac["nmf"], 3) if jac["nmf"] else "—", "Critiques": ap.fr_entier(topics["corpus"]["documents"]),
            },
            "LDA": {
                "Nombre de thèmes": lda["k"], "Cohérence c_v": ap.fr_nombre(lda["c_v"], 3), "Diversité": ap.fr_nombre(lda["diversite"], 3),
                "Stabilité (Jaccard)": ap.fr_nombre(jac["lda"], 3) if jac["lda"] else "—", "Critiques": ap.fr_entier(topics["corpus"]["documents"]),
            },
            "BERTopic (BGE-M3, k-means)": {
                "Nombre de thèmes": reglage["n_themes"], "Cohérence c_v": ap.fr_nombre(reglage["c_v"], 3), "Diversité": ap.fr_nombre(reglage["diversite"], 3),
                "Stabilité (Jaccard)": ap.fr_nombre(reglage["stabilite_jaccard_moyen"], 3), "Critiques": ap.fr_entier(colab["critiques"]),
            },
        }).astype(str))


ui.entete("Modélisation thématique des critiques et comparaison selon la polarité")
methode = st.segmented_control("Méthode", list(METHODES), default="NMF (par défaut)", required=True, key="methode_themes")
if METHODES[methode] == "bertopic":
    afficher_bertopic()
else:
    afficher_cours(METHODES[methode])
afficher_comparaison()

ui.commentaire(artefacts, [
    f"**Ce qui est présenté.** Trois méthodes décrivent les thèmes des critiques : NMF à {topics['choix_k']['nmf']['k']} thèmes (méthode par défaut), "
    f"LDA à {topics['choix_k']['lda']['k']} thèmes (contrôle) et BERTopic, calculé sur Google Colab avec l'encodeur BGE-M3 sur les "
    f"{ap.fr_entier(colab['critiques'])} critiques, regroupées en {colab['reglage_retenu']['n_themes']} thèmes par k-means. "
    "Les thèmes décrivent le corpus ; ils sont ajustés sans les étiquettes de sentiment.",
    "**Comment lire.** L'écart de prévalence est la part moyenne d'un thème dans les avis positifs moins sa part dans les avis négatifs "
    "(barres fines : intervalle de confiance à 95 %). Pour BERTopic, chaque critique appartient à un seul thème et la part est le "
    "pourcentage de critiques de chaque polarité classées dans ce thème. La carte en deux dimensions montre un échantillon de "
    "4 000 critiques tirées au hasard (graine 42).",
    "**Précautions.** Le nom français des 30 thèmes BERTopic est une proposition, à relire (`docs/noms_themes_bertopic.md`). "
    "La stabilité de BERTopic est modérée. Les résultats BERTopic sont précalculés : le modèle ne tourne pas dans cette application. "
    f"Calcul Colab du {colab['date_calcul_utc'][:10]}, source `colab/resultats_colab.zip`.",
])

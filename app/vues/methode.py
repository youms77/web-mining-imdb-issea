"""Page « Méthodologie et limites » : données, prétraitement, vectorisation, modèles, évaluation, limites."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import interface as ui
from src import application as ap

artefacts = ui.charger()
metriques = artefacts.metriques
donnees = metriques["donnees"]
sentiment = metriques["sentiment"]
application = metriques["application"]
seuils = application["seuils_texte"]
zone = sentiment["test"]["zone_incertitude"]
choix_svm = sentiment["choix_figes"]["reglages_par_famille"]["svm_lineaire"]
k_themes = metriques["topics"]["modele_commun"]["k"]
colab = metriques["bertopic_colab"]
separation = metriques["pretraitement"]["separation"]
n_test = separation["test"]

ui.entete("Méthodologie de l'étude et limites des résultats")

ETAPES = [
    ("1. Données", "bleu", [
        f"Corpus « IMDB Dataset of 50K Movie Reviews » (Maas et al., 2011) : {ap.fr_entier(donnees['brut']['lignes'])} critiques en anglais, classes équilibrées.",
        f"Suppression de {ap.fr_entier(donnees['doublons']['sans_ponctuation']['lignes_en_trop'])} doublons : {ap.fr_entier(donnees['corpus_retenu']['lignes'])} critiques retenues.",
        f"Séparation 80/20 stratifiée, graine 42 : {ap.fr_entier(separation['entrainement'])} critiques d'entraînement et {ap.fr_entier(separation['test'])} de test.",
    ]),
    ("2. Prétraitement", "vert", [
        "Retrait des balises HTML et des adresses, mise en minuscules, développement des contractions négatives.",
        "Lemmatisation (WordNet) pour les thèmes, retrait des mots vides et des mots génériques du domaine.",
        "Le même code est utilisé pour l'entraînement, le notebook et l'application.",
    ]),
    ("3. Vectorisation", "orange", [
        "TF-IDF sur mots et paires de mots pour le classement du sentiment ; sac de mots et TF-IDF comparés par validation croisée.",
        "Les vectoriseurs sont ajustés sur l'entraînement seul, dans un `Pipeline`.",
        f"Encodeur BGE-M3 ({colab['encodeur']['licence']}) pour BERTopic, calculé sur Google Colab.",
    ]),
    ("4. Modèles", "bleu", [
        f"Sentiment : Naive Bayes, régression logistique et SVM linéaire (C = {ap.fr_nombre(choix_svm['valeur'], 0)}), calibration sigmoïde pour obtenir une probabilité ; VADER comme référence.",
        f"Thèmes : NMF à {k_themes} thèmes (par défaut), LDA à {metriques['topics']['choix_k']['lda']['k']} thèmes, BERTopic à {colab['reglage_retenu']['n_themes']} thèmes (k-means).",
        "Les thèmes sont descriptifs et ajustés sans les étiquettes de sentiment.",
    ]),
    ("5. Évaluation", "vert", [
        f"Test unique sur {ap.fr_entier(n_test)} critiques, ouvert une seule fois après le gel des choix.",
        "Exactitude avec intervalle de confiance à 95 %, F1, ROC-AUC, matrice de confusion, calibration.",
        f"Zone d'incertitude entre {ap.fr_nombre(zone['seuil_bas'], 2)} et {ap.fr_nombre(zone['seuil_haut'], 2)}, fixée avant d'ouvrir le test.",
    ]),
]
for debut in range(0, len(ETAPES), 3):
    for colonne, (titre, ton, puces) in zip(st.columns(3, gap="large"), ETAPES[debut:debut + 3]):
        with ui.carte(f"etape_{debut}_{titre[0]}", colonne, ton=ton, height="stretch"):
            st.subheader(titre, anchor=False)
            st.markdown("\n".join(f"- {p}" for p in puces))

st.subheader("6. Limites", anchor=False)
col_proba, col_court = st.columns(2, gap="large")
with col_proba:
    with ui.carte("lecture", ton="orange", height="stretch"):
        st.markdown("**Interprétation de la probabilité estimée**")
        st.markdown(
            "Un score de 0,80 signifie : parmi des critiques **entières** ayant ce score, environ 80 % sont positives. "
            "Ce n'est pas la certitude que *cette* critique est positive."
        )
        st.markdown(
            f"Entre {ap.fr_nombre(zone['seuil_bas'], 2)} et {ap.fr_nombre(zone['seuil_haut'], 2)}, l'application affiche "
            f"« incertain ». Sur le test, l'exactitude y tombe à {ap.fr_pct(zone['exactitude_dans_la_zone'])}, contre "
            f"{ap.fr_pct(zone['exactitude_hors_zone'])} en dehors."
        )
with col_court:
    with ui.carte("textes_courts", ton="rouge", height="stretch"):
        st.markdown("**Fiabilité de la prédiction selon la longueur de la critique**")
        st.markdown(
            f"Sous **{seuils['mots_min_fiable']} mots utiles**, le score est signalé comme non fiable. Ce seuil est le "
            f"1er percentile des longueurs d'entraînement : seules {ap.fr_pct(seuils['part_entrainement_sous_le_seuil_fiable'])} "
            "des critiques sont plus courtes."
        )
        lignes = [l for l in application["fiabilite_par_longueur_hors_pli"] if l["mots_max_exclu"] and l["mots_max_exclu"] <= 30]
        if lignes:
            effectif = sum(l["effectif"] for l in lignes)
            confiance = sum(l["confiance_moyenne"] * l["effectif"] for l in lignes) / effectif
            exactitude = sum(l["exactitude"] * l["effectif"] for l in lignes) / effectif
            st.markdown(
                f"Sous 30 mots ({ap.fr_entier(effectif)} critiques, scores hors pli), le modèle annonce en moyenne "
                f"{ap.fr_pct(confiance)} de confiance pour {ap.fr_pct(exactitude)} d'exactitude : il est trop sûr de lui."
            )

st.subheader("Limites d'utilisation de l'application", anchor=False)
limites = [
    ("Avis neutres", "Le corpus ne contient que des avis tranchés (notes ≤ 4 ou ≥ 7). La zone d'incertitude n'a pas été validée sur de vrais avis neutres."),
    ("Autres langues", "Le modèle n'a appris qu'en anglais. Un texte qui ne semble pas anglais est refusé, avec une règle simple qui peut se tromper."),
    ("Sarcasme et second degré", "Le modèle lit des mots, pas l'intention. L'ironie et les critiques mitigées font partie des erreurs relevées."),
    ("Phrases isolées", "Le modèle est calibré sur des critiques entières. Sur une phrase seule, son score peut être très tranché sans être fiable."),
    ("Autres domaines", "Il a appris sur des critiques de films IMDB. Rien ne garantit ses résultats sur des avis de restaurants ou de produits."),
    ("Causalité", "Un mot surligné a pesé dans le score. Cela ne prouve pas que ce mot soit la raison de l'avis."),
]
tons = ["bleu", "vert", "orange", "rouge", "bleu", "vert"]
for i in range(0, len(limites), 3):
    for j, (colonne, (titre, texte)) in enumerate(zip(st.columns(3), limites[i:i + 3])):
        with ui.carte(f"limite_{i + j}", colonne, ton=tons[i + j], height="stretch"):
            st.markdown(f"**{titre}**")
            st.markdown(texte)

with st.expander("Fiabilité par longueur de critique (scores hors pli, entraînement)", icon=":material/table_chart:"):
    table = pd.DataFrame([
        {
            "Mots utiles": (
                f"Moins de {l['mots_max_exclu']}" if l["mots_min"] == 0
                else f"{l['mots_min']} à {l['mots_max_exclu'] - 1}" if l["mots_max_exclu"]
                else f"{l['mots_min']} et plus"
            ),
            "Critiques": ap.fr_entier(l["effectif"]),
            "Exactitude": ap.fr_pct(l["exactitude"]),
            "Confiance moyenne": ap.fr_pct(l["confiance_moyenne"]),
        } for l in application["fiabilite_par_longueur_hors_pli"]
    ]).set_index("Mots utiles")
    st.table(table)

with st.expander("Données, licences et précautions", icon=":material/gavel:"):
    st.markdown(
        "- **Corpus :** *Large Movie Review Dataset* (Maas et al., 2011), version Kaggle. Sa licence de réutilisation reste **[à vérifier]**.\n"
        "- **Ce qui est embarqué :** modèles légers, agrégats et quelques courts extraits. Le corpus complet n'est pas distribué.\n"
        "- **Ressources linguistiques :** WordNet 3.0 (licence Princeton) et étiqueteur NLTK (licence MIT).\n"
        "- **Photographie de l'accueil :** licence libre, voir `app/assets/fonds_candidats/SOURCES.md`.\n"
        "- **Vie privée :** le texte saisi est traité en mémoire et n'est pas enregistré par l'application."
    )

ui.commentaire(artefacts, [
    "**Ce qui est présenté.** La chaîne complète de l'étude, de la collecte au test, puis les limites connues : "
    "avis neutres, autres langues, ironie, phrases isolées, autres domaines et lecture causale des mots.",
    "**Comment lire.** Les cinq blocs colorés suivent l'ordre du travail. Les limites rappellent que la probabilité affichée est calibrée "
    "sur des critiques entières, et que le modèle est moins fiable sur les textes courts.",
])

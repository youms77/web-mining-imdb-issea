"""Page « Méthode et limites » : ce que fait l'application et ce qu'elle ne sait pas faire."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import interface as ui
from src import application as ap

artefacts = ui.charger()
metriques = artefacts.metriques
sentiment = metriques["sentiment"]
application = metriques["application"]
seuils = application["seuils_texte"]
zone = sentiment["test"]["zone_incertitude"]
choix_svm = sentiment["choix_figes"]["reglages_par_famille"]["svm_lineaire"]
k_themes = metriques["topics"]["modele_commun"]["k"]

ui.entete(
    "Ce que l'application fait, et ce qu'elle ne sait pas faire",
    "Un outil d'aide à la lecture de critiques en anglais, entraîné sur un corpus précis. Il ne remplace pas un lecteur.",
)

etapes = [
    (":material/cleaning_services:", "1.\u00a0Nettoyer", "Balises, liens, contractions et notes écrites sont traités par le même code que pendant l'entraînement."),
    (":material/calculate:", "2.\u00a0Pondérer", "TF-IDF sur mots et paires de mots : un mot rare et discriminant pèse plus qu'un mot banal."),
    (":material/tune:", "3.\u00a0Classer et calibrer", f"SVM linéaire (C = {ap.fr_nombre(choix_svm['valeur'], 0)}), puis calibration sigmoïde pour obtenir une probabilité."),
    (":material/category:", "4.\u00a0Situer", f"NMF à {k_themes} thèmes sur un texte lemmatisé. Les parts affichées sont des poids ramenés à 100 %."),
]
for n, (colonne, (icone, titre, texte)) in enumerate(zip(st.columns(4), etapes)):
    with ui.carte(f"etape_{n}", colonne, height="stretch"):
        st.markdown(f"#### :blue[{icone}] {titre}")
        st.markdown(texte)

col_proba, col_court = st.columns(2, gap="large")
with col_proba:
    with ui.carte("lecture", height="stretch"):
        st.subheader("Lire la probabilité", anchor=False)
        st.markdown(
            "Un score de 0,80 signifie : parmi des critiques **entières** ayant ce score, environ 80 % sont positives. "
            "Ce n'est pas la certitude que *cette* critique est positive."
        )
        st.markdown(
            f"Entre {ap.fr_nombre(zone['seuil_bas'], 2)} et {ap.fr_nombre(zone['seuil_haut'], 2)}, l'application affiche "
            f"« incertain ». Sur le test, l'exactitude y tombe à {ap.fr_pct(zone['exactitude_dans_la_zone'])}, contre "
            f"{ap.fr_pct(zone['exactitude_hors_zone'])} en dehors."
        )
with col_court:
    with ui.carte("textes_courts", height="stretch"):
        st.subheader("Pourquoi se méfier des textes courts", anchor=False)
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

st.subheader("Ce qu'elle ne sait pas faire", anchor=False)
limites = [
    (":material/sentiment_neutral:", "Avis neutres", "Le corpus ne contient que des avis tranchés (notes ≤ 4 ou ≥ 7). La zone d'incertitude n'a pas été validée sur de vrais avis neutres."),
    (":material/translate:", "Autres langues", "Le modèle n'a appris qu'en anglais. Un texte qui ne semble pas anglais est refusé, avec une règle simple qui peut se tromper."),
    (":material/theater_comedy:", "Sarcasme et second degré", "Le modèle lit des mots, pas l'intention. L'ironie et les critiques mitigées font partie des erreurs relevées."),
    (":material/short_text:", "Phrases isolées", "Le modèle est calibré sur des critiques entières. Sur une phrase seule, son score peut être très tranché sans être fiable."),
    (":material/movie_filter:", "Autres domaines", "Il a appris sur des critiques de films IMDB. Rien ne garantit ses résultats sur des avis de restaurants ou de produits."),
    (":material/link_off:", "Causalité", "Un mot surligné a pesé dans le score. Cela ne prouve pas que ce mot soit la raison de l'avis."),
]
for i in range(0, len(limites), 3):
    for j, (colonne, (icone, titre, texte)) in enumerate(zip(st.columns(3), limites[i:i + 3])):
        with ui.carte(f"limite_{i + j}", colonne, height="stretch"):
            st.markdown(f"**:orange[{icone}] {titre}**")
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
        "- **Vie privée :** le texte saisi est traité en mémoire et n'est pas enregistré par l'application."
    )

ui.pied(artefacts)

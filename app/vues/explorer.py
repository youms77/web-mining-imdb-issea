"""Page « Explorer les thèmes » : résultats précalculés de NMF, LDA et BERTopic."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

import interface as ui
from src import application as ap

artefacts = ui.charger()
metriques = artefacts.metriques
topics = metriques["topics"]
bertopic = metriques["bertopic"]
extraits = {e["topic"]: e for e in artefacts.contenu.get("extraits", [])}
METHODES = {"NMF (retenu)": "nmf", "LDA": "lda", "BERTopic": "bertopic"}
NATURES = {"sujet": "Sujet", "jugement": "Jugement", "mixte": "Mixte"}


def afficher_fiche(cle: str) -> None:
    """Fiche d'un thème NMF ou LDA : mots, présence par polarité et extrait éventuel."""
    fiches = {f["topic"]: f for f in topics["fiches_topics"][cle]}
    polarite = {l["topic"]: l for l in topics["comparaison_polarite"][cle]["topics"]}
    ordre = [l["topic"] for l in topics["comparaison_polarite"][cle]["topics"]]
    choix = st.selectbox("Choisir un thème", ordre, format_func=lambda t: fiches[t]["nom"], key=f"fiche_{cle}")
    fiche, ligne = fiches[choix], polarite[choix]
    with ui.carte(f"fiche_{cle}"):
        st.subheader(fiche["nom"], anchor=False)
        st.markdown(f"**Nature :** {NATURES.get(fiche['nature'], fiche['nature'])}")
        st.markdown(f"**Mots principaux :** {', '.join(fiche['mots'])}")
        st.markdown(fiche["lecture"])
        with st.container(horizontal=True, gap="medium"):
            st.metric("Avis positifs", f"{ap.fr_nombre(ligne['prevalence_positif_pct'], 1)} %", width="content")
            st.metric("Avis négatifs", f"{ap.fr_nombre(ligne['prevalence_negatif_pct'], 1)} %", width="content")
            st.metric("Écart", ui.fr_ecart(ligne["ecart_points"]), width="content")
        st.caption("Part moyenne du thème dans chaque groupe d'avis.")
        st.caption(
            f"Intervalle de confiance à 95 % de l'écart : {ap.fr_nombre(ligne['ic95_bas'], 1)} à "
            f"{ap.fr_nombre(ligne['ic95_haut'], 1)} points. d de Cohen : {ap.fr_nombre(ligne['d_cohen'], 2)}."
        )
        extrait = extraits.get(choix) if cle == "nmf" else None
        if extrait and extrait.get("extrait"):
            st.markdown("**Extrait d'une critique** (texte nettoyé)")
            st.html(f'<div class="extrait">{escape(extrait["extrait"])}</div>')


def afficher_cours(cle: str) -> None:
    comparaison = topics["comparaison_polarite"][cle]
    lignes = comparaison["topics"]
    fort = max(lignes, key=lambda l: abs(l["ecart_points"]))
    choix_k = topics["choix_k"][cle]
    if cle == "nmf":
        st.success(
            f"Modèle commun du projet : NMF à {choix_k['k']} thèmes, sur {ap.fr_entier(topics['corpus']['documents'])} critiques.",
            icon=":material/verified:",
        )
    else:
        st.info(
            f"LDA à {choix_k['k']} thèmes, calculée comme contrôle du modèle NMF. Elle garde un thème flou à la lecture.",
            icon=":material/compare_arrows:",
        )
    a, b, c = st.columns(3)
    a.metric("Thèmes", choix_k["k"], border=True)
    b.metric("Cohérence c_v", ap.fr_nombre(choix_k["c_v"], 3), border=True)
    c.metric("Diversité des mots", ap.fr_nombre(choix_k["diversite"], 3), border=True)

    col_graphe, col_fiche = st.columns([1.1, 1], gap="large")
    with col_graphe:
        with ui.carte(f"polarite_{cle}"):
            signe = "positifs" if fort["ecart_points"] > 0 else "négatifs"
            st.subheader(f"« {fort['nom']} » sépare le plus les avis", anchor=False)
            st.markdown(f"Ce thème pèse {ap.fr_nombre(abs(fort['ecart_points']), 1)} points de plus dans les avis {signe}.")
            ui.afficher(ui.barres_divergentes(lignes), f"polarite_{cle}", "Écart de présence de chaque thème entre avis positifs et négatifs.")
            st.caption("Part moyenne du thème dans les avis positifs moins les avis négatifs. Barres fines : IC à 95 %.")
    with col_fiche:
        afficher_fiche(cle)


def afficher_bertopic() -> None:
    concentration = bertopic["concentration"]["bertopic_k8"]
    comparaison = bertopic["comparaison"]
    st.warning(
        f"BERTopic a été ajusté sur {ap.fr_entier(bertopic['echantillon']['critiques'])} critiques d'entraînement, pas sur tout "
        f"le corpus. Il laisse {ap.fr_pct(concentration['part_hors_topic'])} des critiques hors topic. "
        "**NMF reste le modèle commun.** Résultats précalculés : BERTopic ne tourne pas dans cette application.",
        icon=":material/warning:",
    )
    a, b, c = st.columns(3)
    a.metric("Critiques hors topic", ap.fr_pct(concentration["part_hors_topic"]), border=True)
    b.metric("Classées dans le plus grand topic", ap.fr_pct(concentration["part_du_plus_grand_topic_parmi_les_classees"]), border=True)
    c.metric(
        "Topics de 10 critiques ou moins",
        f"{concentration['topics_de_10_critiques_ou_moins']} sur {comparaison['bertopic_k8']['n_topics']}", border=True,
    )

    col_tableau, col_graphe = st.columns([1, 1.1], gap="large")
    with col_tableau:
        with ui.carte("bertopic_comparaison"):
            st.subheader("Comparaison sur le même échantillon", anchor=False)
            modeles = {"BERTopic k=8": "bertopic_k8", "NMF k=8": "nmf_k8_echantillon", "LDA k=12": "lda_k12_echantillon"}
            st.table(pd.DataFrame({
                nom: {
                    "Cohérence c_v": ap.fr_nombre(comparaison[c]["c_v"], 3),
                    "Diversité": ap.fr_nombre(comparaison[c]["diversite"], 3),
                    "Hors topic": ap.fr_pct(comparaison[c]["part_hors_topic"]),
                    "Stabilité (Jaccard)": ap.fr_nombre(comparaison[c]["stabilite_jaccard"], 3),
                } for nom, c in modeles.items()
            }))
            st.markdown(bertopic["conclusion"])
    with col_graphe:
        with ui.carte("bertopic_polarite"):
            st.subheader("Topics BERTopic par polarité", anchor=False)
            lignes = [l for l in bertopic["polarite_k8"]["topics"] if l["topic"] >= 0]
            ui.afficher(ui.barres_divergentes(lignes), "polarite_bertopic", "Écart de présence des topics BERTopic entre avis positifs et négatifs.")
            st.caption(
                f"{concentration['topics_de_10_critiques_ou_moins']} topics sur {len(lignes)} comptent 10 critiques "
                "ou moins : leurs intervalles sont très larges."
            )

    phrases = bertopic["extension_phrases"]
    with st.expander("Extension exploratoire : thèmes et sentiment par phrase", icon=":material/science:"):
        st.warning(phrases["avertissement"], icon=":material/science:")
        mixtes = phrases["critiques_mixtes"]["seuils_figes"]
        x, y, z = st.columns(3)
        x.metric("Phrases scorées", ap.fr_entier(phrases["phrases_scorees"]))
        y.metric("Phrases hors aspect", ap.fr_pct(phrases["part_phrases_hors_aspect"]))
        z.metric("Avis positifs à phrase négative", ap.fr_pct(mixtes["critiques_positives_avec_une_phrase_negative"]))
        st.markdown(
            f"Accord modeste avec VADER sur les phrases : {ap.fr_pct(phrases['accord_vader']['accord_brut_vader_non_neutre'])}, "
            f"kappa {ap.fr_nombre(phrases['accord_vader']['kappa_cohen_vader_non_neutre'], 3)}. VADER n'est pas une vérité de référence."
        )


ui.entete(
    "De quoi parlent les critiques ?",
    "Comparez les thèmes des avis favorables et défavorables, avec trois méthodes. Tous les résultats sont précalculés.",
)
methode = st.segmented_control("Méthode", list(METHODES), default="NMF (retenu)", required=True, key="methode_themes")
if METHODES[methode] == "bertopic":
    afficher_bertopic()
else:
    afficher_cours(METHODES[methode])
ui.pied(artefacts, "Thèmes descriptifs, ajustés sans les étiquettes.")

"""Page « Analyse descriptive » : taille, classes, longueur, mots et bruit du corpus."""
from __future__ import annotations

import streamlit as st

import interface as ui
from src import application as ap

artefacts = ui.charger()
m = artefacts.metriques
donnees = m["donnees"]
agr = ui.agregats()
retenu = donnees["corpus_retenu"]
bruit = donnees["bruit"]
notes = donnees["notes_dans_le_texte"]
long = donnees["longueurs"]
distinctifs = m["topics"]["mots_distinctifs"]

ui.entete("Analyse descriptive du corpus de critiques IMDB")

ui.indicateurs([
    ("Critiques brutes", ap.fr_entier(donnees["brut"]["lignes"]), "bleu"),
    ("Critiques retenues", ap.fr_entier(retenu["lignes"]), "vert", f"{ap.fr_entier(donnees['doublons']['sans_ponctuation']['lignes_en_trop'])} doublons retirés"),
    ("Médiane de mots par critique", f"{ap.fr_entier(long['negative']['mots']['mediane'])} · {ap.fr_entier(long['positive']['mots']['mediane'])}", "orange", "avis négatifs · avis positifs"),
    ("Valeurs manquantes", ap.fr_entier(sum(donnees["brut"]["manquants"].values())), "bleu"),
])

col_a, col_b = st.columns(2, gap="large")
with col_a:
    with ui.carte("classes", ton="vert"):
        st.subheader("Répartition des classes", anchor=False)
        ui.afficher(ui.barres_classes(retenu["classes"]), "classes", "Nombre de critiques négatives et positives du corpus retenu.")
with col_b:
    with ui.carte("longueurs", ton="orange"):
        st.subheader("Distribution de la longueur des critiques par classe", anchor=False)
        ui.afficher(ui.histogramme_longueurs(agr["histogramme_longueurs"]), "longueurs", "Nombre de critiques par tranche de 25 mots, selon la classe.")

st.subheader("Mots les plus associés à chaque polarité", anchor=False)
col_neg, col_pos = st.columns(2, gap="large")
with col_neg:
    with ui.carte("mots_neg", ton="rouge"):
        st.markdown("**Avis négatifs**")
        ui.afficher(ui.barres_mots(distinctifs["plus_negatifs"][:15], ui.ROUGE), "mots_neg", "Quinze mots les plus associés aux avis négatifs.")
with col_pos:
    with ui.carte("mots_pos", ton="vert"):
        st.markdown("**Avis positifs**")
        ui.afficher(ui.barres_mots(distinctifs["plus_positifs"][:15], ui.VERT), "mots_pos", "Quinze mots les plus associés aux avis positifs.")

with ui.carte("bruit"):
    st.subheader("Balises HTML et notes écrites dans le texte", anchor=False)
    ui.afficher(ui.barres_parts([
        ("Balises de saut de ligne (br)", bruit["balise_br"]["part_pct"]),
        ("Notes écrites (par exemple 7/10)", notes["part_pct"]),
        ("Adresses web", bruit["url"]["part_pct"]),
        ("Autres balises HTML", bruit["autre_balise"]["part_pct"]),
        ("Entités HTML", bruit["entite_html"]["part_pct"]),
    ]), "bruit", "Part des critiques brutes contenant des balises HTML, des adresses ou une note écrite.")

ui.commentaire(artefacts, [
    f"**Ce qui est présenté.** Le corpus compte {ap.fr_entier(donnees['brut']['lignes'])} critiques brutes ; "
    f"{ap.fr_entier(retenu['lignes'])} sont retenues après suppression de {ap.fr_entier(donnees['doublons']['sans_ponctuation']['lignes_en_trop'])} doublons. "
    "Les deux classes sont presque à parité. La longueur est exprimée en mots, après retrait des balises de saut de ligne.",
    "**Comment lire les graphiques.** La répartition compare les effectifs des deux classes. L'histogramme superpose la longueur "
    "des critiques négatives (rouge) et positives (vert), tronquée à 900 mots. Les mots associés sont classés par "
    "|log-rapport de cotes| (méthode de Monroe, Colaresi et Quinn, 2008) : plus la barre est longue, plus le mot distingue la polarité. "
    "Une barre de bruit donne la part des critiques concernées par chaque type de balise ou de note.",
    f"**Notes écrites.** {ap.fr_nombre(notes['part_pct'], 2)} % des critiques contiennent une note du type « 7/10 » ; "
    f"elle est cohérente avec l'étiquette dans {ap.fr_nombre(notes['accord_pct'], 2)} % des cas où elle est nette.",
    "**Calcul.** Les graphiques utilisent des agrégats précalculés (`models/deploy/agregats_refonte.json`) : l'application "
    "n'ouvre jamais le corpus brut.",
])

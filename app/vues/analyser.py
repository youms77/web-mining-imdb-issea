"""Page « Analyser une critique » : sentiment, incertitude, mots influents et thèmes."""
from __future__ import annotations

from html import escape

import streamlit as st

import interface as ui
from src import application as ap

artefacts = ui.charger()
exemples = artefacts.contenu["exemples"]
metriques = artefacts.metriques
zone_test = metriques["sentiment"]["test"]["zone_incertitude"]
LIBELLES = {"positif": "Exemple positif", "negatif": "Exemple négatif", "mitige": "Exemple mitigé"}
ICONES = {"positif": ":material/thumb_up:", "negatif": ":material/thumb_down:", "mitige": ":material/balance:"}
VERDICTS = {
    "positif": (":green[:material/sentiment_satisfied:]", "Avis positif"),
    "negatif": (":red[:material/sentiment_dissatisfied:]", "Avis négatif"),
    "incertain": (":orange[:material/help:]", "Incertain"),
}


def charger_exemple(cle: str) -> None:
    st.session_state["texte_critique"] = exemples[cle]["texte"]
    st.session_state["exemple_actif"] = cle
    st.session_state["resultat"] = ap.analyser(exemples[cle]["texte"], artefacts)


def lancer_analyse() -> None:
    texte = st.session_state.get("texte_critique", "")
    actif = st.session_state.get("exemple_actif")
    if actif and texte != exemples[actif]["texte"]:
        st.session_state["exemple_actif"] = None
    st.session_state["resultat"] = ap.analyser(texte, artefacts)


def phrase_calibration(p: float) -> str:
    """Ce que le test a observé pour des critiques entières ayant un score voisin."""
    for tranche in metriques["sentiment"]["test"]["fiabilite_modele_retenu"]:
        bas, haut = tranche["tranche"]
        if (bas <= p < haut or (haut == 1.0 and p == 1.0)) and tranche["effectif"] > 0:
            return (
                f"Sur le test, {ap.fr_entier(tranche['effectif'])} critiques entières avaient un score entre "
                f"{ap.fr_nombre(bas, 1)} et {ap.fr_nombre(haut, 1)} : {ap.fr_pct(tranche['frequence_observee'])} "
                "étaient réellement positives."
            )
    return ""


def afficher_verdict(r: dict) -> None:
    with ui.carte("verdict", ton={"positif": "vert", "negatif": "rouge", "incertain": "orange"}.get(r["categorie"], "bleu") if r["fiable"] else "orange"):
        if not r["fiable"]:
            st.markdown("### :orange[:material/warning:] Score non fiable")
            st.warning(r["avertissement"], icon=":material/short_text:")
            st.markdown(
                f"Score brut, à titre indicatif seulement : **{ui.fr_proba(r['probabilite_positive'])}**. "
                "Pour un résultat interprétable, collez la critique entière. Les mots influents et les thèmes "
                "ne sont pas affichés pour un texte aussi court."
            )
            return
        icone, libelle = VERDICTS[r["categorie"]]
        st.markdown(f"### {icone} {libelle}")
        st.metric("Probabilité calibrée que l'avis soit positif", ui.fr_proba(r["probabilite_positive"]))
        ui.afficher(
            ui.jauge(r["probabilite_positive"], r["seuil_bas"], r["seuil_haut"]), "jauge",
            f"Jauge : score {ui.fr_proba(r['probabilite_positive'])}, zone d'incertitude entre "
            f"{ap.fr_nombre(r['seuil_bas'], 2)} et {ap.fr_nombre(r['seuil_haut'], 2)}.",
        )
        if r["categorie"] == "incertain":
            st.warning(
                f"Le modèle ne tranche pas. Dans cette zone, il n'a raison que dans "
                f"{ap.fr_pct(zone_test['exactitude_dans_la_zone'])} des cas sur le test, contre "
                f"{ap.fr_pct(zone_test['exactitude_hors_zone'])} en dehors.",
                icon=":material/help:",
            )
        st.markdown(phrase_calibration(r["probabilite_positive"]))


def afficher_mots(r: dict) -> None:
    with ui.carte("mots"):
        st.subheader("Contribution des mots à la prédiction", anchor=False)
        st.html(
            '<div class="legende"><span><mark class="mot-pos">souligné plein</mark> pousse vers positif</span>'
            '<span><mark class="mot-neg">pointillé</mark> pousse vers négatif</span></div>'
            f'<div class="texte-analyse">{r["html_surligne"]}</div>'
        )


def afficher_themes(r: dict) -> None:
    with ui.carte("themes"):
        st.subheader("Thèmes les plus proches de la critique", anchor=False)
        if not r["themes"]:
            st.info("Aucun mot de ce texte n'appartient au vocabulaire des thèmes.", icon=":material/info:")
            return
        lignes = "".join(
            f'<div class="theme-ligne"><div class="theme-tete"><span>{escape(t["nom"])}</span>'
            f'<span>{ap.fr_nombre(t["part_pct"], 1)}&nbsp;%</span></div>'
            f'<div class="theme-barre"><span style="width:{t["part_pct"]:.1f}%"></span></div>'
            f'<div class="theme-mots">{escape(", ".join(t["mots"]))}</div></div>'
            for t in r["themes"]
        )
        st.html(lignes)


ui.entete("Prédiction du sentiment et des thèmes d'une nouvelle critique")

col_saisie, col_resultat = st.columns([1.08, 1], gap="large")
with col_saisie:
    with ui.carte("saisie", ton="bleu"):
        st.markdown("**Essayer un exemple**")
        with st.container(horizontal=True, gap="small"):
            for cle in LIBELLES:
                st.button(
                    LIBELLES[cle].removeprefix("Exemple ").capitalize(), key=f"exemple_{cle}", icon=ICONES[cle],
                    on_click=charger_exemple, args=(cle,), width="content",
                )
        st.text_area(
            "Votre critique (en anglais)", key="texte_critique", height=210,
            placeholder="Paste a full movie review here…",
        )
        st.button("Analyser", key="analyser", type="primary", icon=":material/search:", on_click=lancer_analyse, width="stretch")

resultat = st.session_state.get("resultat")
with col_resultat:
    if resultat is None:
        with ui.carte("attente"):
            st.markdown("### :blue[:material/movie:] Résultat de la prédiction")
            st.markdown("Choisissez un exemple ou collez une critique entière, puis lancez l'analyse.")
    elif resultat["statut"] == "erreur":
        st.error(resultat["message"], icon=":material/error:")
    else:
        afficher_verdict(resultat)
        if st.session_state.get("exemple_actif") == "mitige":
            info = metriques["application"]["exemple_mitige"]
            st.info(
                f"Avis partagé, classé {info['classe_reelle']} dans le corpus. Le modèle gelé a vu cette critique à "
                f"l'entraînement et annonce {ap.fr_nombre(info['probabilite_modele_gele'], 3)}. Un modèle qui ne l'avait "
                f"pas vue annonçait {ap.fr_nombre(info['probabilite_hors_pli'], 3)}, soit une vraie hésitation.",
                icon=":material/balance:",
            )

if resultat is not None and resultat["statut"] == "ok" and resultat["fiable"]:
    col_mots, col_themes = st.columns([1.5, 1], gap="large")
    with col_mots:
        afficher_mots(resultat)
    with col_themes:
        afficher_themes(resultat)

ui.commentaire(artefacts, [
    "**Ce qui est présenté.** L'application estime le sentiment d'une critique de film en anglais avec le SVM linéaire calibré retenu, "
    "indique la zone d'incertitude, surligne les mots qui ont le plus pesé dans le score et situe le texte parmi les "
    f"{metriques['topics']['modele_commun']['k']} thèmes du modèle NMF.",
    "**Comment lire.** La probabilité est celle que l'avis soit positif. La jauge est rouge (négatif), orange (zone d'incertitude) puis verte (positif). "
    "Un mot souligné en trait plein pousse vers positif, en pointillé vers négatif : ce sont les poids du modèle linéaire avant calibration, "
    "sans valeur d'explication causale. Les parts des thèmes sont des poids ramenés à 100 %, pas des probabilités.",
    "**Précautions.** Les exemples proposés servent à essayer l'outil ; l'exemple mitigé a été vu à l'entraînement par le modèle gelé. "
    "Un texte trop court ou qui ne semble pas anglais n'est pas analysé de façon fiable.",
])

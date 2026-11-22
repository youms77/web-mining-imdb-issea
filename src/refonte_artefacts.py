"""Refonte de l'application : prépare des agrégats légers dans `models/deploy/agregats_refonte.json`.

Ce script se lance une fois, en local. L'application n'ouvre jamais le corpus brut :
elle lit seulement ce fichier léger et `reports/metrics.json`.

Contenu produit :
- histogramme de la longueur des critiques par classe (corpus dédoublonné) ;
- courbes ROC des 5 modèles, et exactitude du SVM par tranche de probabilité, calculées à partir
  des prédictions de test DÉJÀ enregistrées (aucun modèle appelé, aucun réentraînement) ;
- résultats BERTopic calculés sur Colab (BGE-M3, 49 575 critiques, 30 thèmes), lus dans
  `data/interim/colab/` (archive extraite, traitée comme des données).

Il ajoute à `metrics.json` seulement de nouvelles clés (`bertopic_colab`, `refonte_agregats`) et
refuse d'écrire si une clé existante a changé.

Exemple d'utilisation :
    .\\.venv\\Scripts\\python.exe -m src.refonte_artefacts
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.data import charger_brut, cle_normalisee

RACINE = Path(__file__).resolve().parents[1]
COLAB = RACINE / "data" / "interim" / "colab"
PHASE5 = RACINE / "data" / "interim" / "phase5"
SORTIE = RACINE / "models" / "deploy" / "agregats_refonte.json"
METRIQUES = RACINE / "reports" / "metrics.json"
GRAINE = 42
MODELES_ROC = ["naive_bayes_bow", "naive_bayes_tfidf", "regression_logistique", "svm_lineaire", "vader_compound"]
PAS_MOTS, MAX_MOTS = 25, 900
NB_POINTS_CARTE = 4000

# Noms français PROPOSÉS à partir des mots de chaque thème (à relire : voir docs/noms_themes_bertopic.md).
NOMS_THEMES = {
    0: "Jeu d'acteurs et distribution", 1: "Rôles féminins et vie de couple", 2: "Séries télévisées et sitcoms",
    3: "Policier, crime et enquêtes", 4: "Intrigue, scénario et personnages", 5: "Cinéma français et européen",
    6: "Jugements très négatifs (déception, argent perdu)", 7: "Achat et édition en DVD et VHS",
    8: "Science-fiction et extraterrestres", 9: "Rôles d'actrices et personnages féminins",
    10: "Films d'animation et dessins animés", 11: "Vie, amour et relations", 12: "Horreur à petit budget",
    13: "Documentaires et religion", 14: "Comédie et humour", 15: "Guerre et Seconde Guerre mondiale",
    16: "Cinéma asiatique et arts martiaux", 17: "Horreur : tueurs et monstres", 18: "Westerns et histoire américaine",
    19: "Zombies et horreur gore", 20: "Adaptations de romans et classiques", 21: "Films d'action",
    22: "Comédies musicales et danse", 23: "Animation japonaise et jeux", 24: "Comédies classiques et duos comiques",
    25: "Musique, concerts et groupes", 26: "Horreur italienne (giallo)", 27: "Séries de science-fiction et catch",
    28: "Cinéma indien (Bollywood)", 29: "Films de guerre et d'action militaire",
}


def histogramme_longueurs(ids_retenus: set[int]) -> dict:
    """Nombre de critiques par tranche de 25 mots, par classe, sur le corpus dédoublonné."""
    brut = charger_brut()
    brut = brut[brut["id_brut"].isin(ids_retenus)]
    mots = brut["review"].map(cle_normalisee).str.split().str.len()
    bornes = list(range(0, MAX_MOTS + PAS_MOTS, PAS_MOTS))
    sortie = {"pas_mots": PAS_MOTS, "maximum_affiche": MAX_MOTS, "bornes": bornes[:-1], "classes": {}}
    for classe in ("negative", "positive"):
        valeurs = mots[brut["sentiment"] == classe]
        comptes, _ = np.histogram(valeurs, bins=bornes)
        sortie["classes"][classe] = {
            "effectifs": [int(c) for c in comptes],
            "au_dela": int((valeurs > MAX_MOTS).sum()),
            "total": int(len(valeurs)),
        }
    return sortie


def courbes_roc() -> dict:
    """ROC des 5 modèles sur le test, à partir des prédictions déjà enregistrées."""
    pred = pd.read_parquet(PHASE5 / "predictions_test.parquet")
    etiq = pd.read_parquet(PHASE5 / "etiquettes_test.parquet")
    donnees = pred.merge(etiq, on="id_brut", validate="one_to_one")
    y = donnees["y"].to_numpy()
    sortie = {}
    for nom in MODELES_ROC:
        score = donnees[nom].to_numpy()
        fpr, tpr, _ = roc_curve(y, score)
        pas = max(1, len(fpr) // 250)  # courbe allégée, forme conservée
        indices = sorted(set(range(0, len(fpr), pas)) | {len(fpr) - 1})
        sortie[nom] = {
            "auc": float(roc_auc_score(y, score)),
            "fpr": [round(float(fpr[i]), 5) for i in indices],
            "tpr": [round(float(tpr[i]), 5) for i in indices],
        }
    return {"critiques_test": int(len(y)), "modeles": sortie}


def exactitude_par_probabilite(seuil_bas: float, seuil_haut: float) -> dict:
    """Exactitude du SVM calibré par tranche de probabilité (pas de 0,05), sur le test."""
    pred = pd.read_parquet(PHASE5 / "predictions_test.parquet")
    etiq = pd.read_parquet(PHASE5 / "etiquettes_test.parquet")
    donnees = pred.merge(etiq, on="id_brut", validate="one_to_one")
    p = donnees["svm_lineaire"].to_numpy()
    correct = ((p >= 0.5).astype(int) == donnees["y"].to_numpy())
    bornes = np.round(np.arange(0, 1.0001, 0.05), 2)
    indices = np.clip(np.digitize(p, bornes[1:-1]), 0, len(bornes) - 2)
    tranches = []
    for k in range(len(bornes) - 1):
        masque = indices == k
        n = int(masque.sum())
        tranches.append({
            "bas": float(bornes[k]), "haut": float(bornes[k + 1]), "effectif": n,
            "corrects": int(correct[masque].sum()),
        })
    return {"seuil_bas": seuil_bas, "seuil_haut": seuil_haut, "tranches": tranches}


def bertopic_colab(ids_polarite: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Résumé honnête du réglage retenu sur Colab, et points de la carte 2D."""
    resultats = json.loads((COLAB / "resultats.json").read_text(encoding="utf-8"))
    attr = pd.read_parquet(COLAB / "attributions.parquet").merge(ids_polarite, on="id_brut", validate="one_to_one")
    choix = resultats["choix"]
    n_pos, n_neg = int((attr["sentiment"] == "positive").sum()), int((attr["sentiment"] == "negative").sum())
    themes, lignes = [], []
    for t in choix["topics"]:
        pos, neg = t["avis_positifs"], t["avis_negatifs"]
        # Contrôle : les effectifs de Colab doivent égaler ceux recomptés ici.
        sous = attr[attr["topic"] == t["topic"]]
        assert int((sous["sentiment"] == "positive").sum()) == pos and int((sous["sentiment"] == "negative").sum()) == neg
        p1, p0 = pos / n_pos, neg / n_neg
        ecart = 100 * (p1 - p0)
        es = 100 * float(np.sqrt(p1 * (1 - p1) / n_pos + p0 * (1 - p0) / n_neg))
        themes.append({
            "topic": t["topic"], "nom": NOMS_THEMES[t["topic"]], "mots": t["mots"], "effectif": t["effectif"],
            "part_corpus_pct": round(100 * t["part_corpus"], 2), "avis_positifs": pos, "avis_negatifs": neg,
            "c_v": round(t["c_v"], 4),
        })
        lignes.append({
            "topic": t["topic"], "nom": NOMS_THEMES[t["topic"]], "prevalence_positif_pct": round(100 * p1, 3),
            "prevalence_negatif_pct": round(100 * p0, 3), "ecart_points": round(ecart, 3),
            "ic95_bas": round(ecart - 1.96 * es, 3), "ic95_haut": round(ecart + 1.96 * es, 3),
        })
    hdb = [g for g in resultats["grille"] if g["famille"] == "hdbscan"]
    km = [g for g in resultats["grille"] if g["famille"] == "kmeans"]
    resume = {
        "statut": resultats["statut"], "date_calcul_utc": resultats["date_utc"],
        "source": "colab/resultats_colab.zip (résultats.json, attributions.parquet), extrait dans data/interim/colab/",
        "critiques": resultats["corpus"]["critiques"], "empreinte_entree": resultats["corpus"]["empreinte_entree"],
        "encodeur": {k: resultats["encodeur"][k] for k in ("nom", "revision", "max_jetons", "pooling", "licence")},
        "materiel": resultats["materiel"], "versions": resultats["versions"],
        "reglage_retenu": {
            "identifiant": choix["reglage"], "methode": "k-means", "n_themes": choix["n_topics"],
            "texte": "sans les mots génériques (movie, film, watch…)", "reduction": "ACP à 50 axes, axe 1 retiré",
            "raison_axe_1": "l'axe 1 est fortement lié au sentiment (corrélation "
                            f"{choix['correlation_axe1_sentiment']:.2f}) ; le retirer évite de regrouper les critiques par polarité",
            "part_hors_theme": choix["part_hors_topic"], "plus_grand_theme_part": choix["plus_grand_topic"],
            "diversite": choix["diversite"], "c_v": choix["c_v"], "c_npmi": choix["c_npmi"],
            "stabilite_jaccard_moyen": choix["stabilite"]["jaccard_moyen"],
            "stabilite_par_graine": choix["stabilite"]["par_graine"],
            "stabilite_part_sous_echantillon": choix["stabilite"]["part_sous_echantillon"],
        },
        "regle_de_choix": resultats["regle"]["texte"],
        "grille": {
            "reglages_evalues": len(resultats["grille"]), "reglages_kmeans": len(km),
            "reglages_hdbscan": len(hdb), "hdbscan_themes_max": max(g["n_topics"] for g in hdb),
            "hdbscan_plus_grand_theme_min": min(g["plus_grand_topic"] for g in hdb),
        },
        "polarite": {"avis_positifs": n_pos, "avis_negatifs": n_neg, "themes": lignes},
        "themes": themes, "durees_s": resultats["durees_s"],
        "noms_themes": "propositions en français à partir des mots de chaque thème, à relire (docs/noms_themes_bertopic.md)",
    }
    graine = attr.sample(NB_POINTS_CARTE, random_state=GRAINE).sort_values("id_brut")
    carte = [
        {"x": round(float(r.x), 3), "y": round(float(r.y), 3), "t": int(r.topic), "s": 1 if r.sentiment == "positive" else 0}
        for r in graine.itertuples()
    ]
    return resume, carte


def main() -> None:
    metriques = json.loads(METRIQUES.read_text(encoding="utf-8"))
    textes = pd.read_parquet(RACINE / "data" / "interim" / "textes_pretraites.parquet", columns=["id_brut", "sentiment"])
    zone = metriques["sentiment"]["test"]["zone_incertitude"]
    colab, carte = bertopic_colab(textes)
    agregats = {
        "histogramme_longueurs": histogramme_longueurs(set(textes["id_brut"])),
        "roc": courbes_roc(),
        "incertitude": exactitude_par_probabilite(zone["seuil_bas"], zone["seuil_haut"]),
    }
    SORTIE.write_text(json.dumps({**agregats, "carte_colab": carte}, ensure_ascii=False), encoding="utf-8")

    nouveau = dict(metriques)
    nouveau["bertopic_colab"] = colab
    nouveau["refonte_agregats"] = {
        "note": "Agrégats calculés par src/refonte_artefacts.py ; courbes tracées dans models/deploy/agregats_refonte.json",
        "auc_recalculee_test": {k: round(v["auc"], 6) for k, v in agregats["roc"]["modeles"].items()},
        "histogramme_longueurs": agregats["histogramme_longueurs"],
        "exactitude_par_probabilite": agregats["incertitude"],
    }
    assert all(nouveau[k] == metriques[k] for k in metriques), "une clé existante a changé"
    METRIQUES.write_text(json.dumps(nouveau, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Agrégats écrits :", SORTIE.name, "| clés ajoutées : bertopic_colab, refonte_agregats")


if __name__ == "__main__":
    main()

"""Phase 6 : polarité des topics BERTopic, poids, figures 17 à 19 et clé `bertopic` de metrics.json.

Le module relit les caches. Il n'écrit que la clé `bertopic` et refuse de continuer si une autre
clé de `metrics.json` a changé.

Exemple d'utilisation :
    python -m src.bertopic_bilan
"""
from __future__ import annotations

import importlib.metadata as im
import json
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from src import bertopic_model as bm
from src import bertopic_phrases as bp
from src import exploration as ex
from src import topics as tp
from src.preprocessing import RACINE

CHEMIN_METRIQUES = RACINE / "reports" / "metrics.json"
CHEMIN_NOMS = RACINE / "src" / "ressources" / "bertopic_noms.json"
CHEMIN_POLARITE = bm.DOSSIER / "polarite_k8.json"
CHEMIN_POIDS = bm.DOSSIER / "poids.json"
COULEUR_RETENU, COULEUR_NEUTRE, COULEUR_INCERTAIN = "#6250d6", "#8a897f", "#c3c2b7"
SOURCE = "Source : IMDB Dataset (Kaggle), échantillon d'entraînement, calculs du projet"
ORDRE_FIGURE = (
    ("nmf_k8_echantillon", "NMF k=8"), ("lda_k12_echantillon", "LDA k=12"),
    ("bertopic_natif", "BERTopic natif"), ("bertopic_k8", "BERTopic k=8"), ("bertopic_k12", "BERTopic k=12"),
)


def _fr(valeur: float, decimales: int = 3) -> str:
    return f"{valeur:.{decimales}f}".replace(".", ",")


def charger_noms(cle: str, mots: list[list[str]]) -> list[dict]:
    """Noms de lecture, avec contrôle que l'ordre des topics n'a pas changé."""
    noms = json.loads(CHEMIN_NOMS.read_text(encoding="utf-8"))[cle]
    if len(noms) != len(mots) or any(n["signature"] != m[0] for n, m in zip(noms, mots)):
        raise RuntimeError(f"Les noms de `{cle}` ne correspondent plus aux topics : relire les topics.")
    return noms


def concentration(sujets: np.ndarray) -> dict:
    """Couverture et concentration d'une partition, topic −1 à part."""
    classes = sujets[sujets != -1]
    tailles = np.bincount(classes)
    tailles = tailles[tailles > 0]
    return {
        "part_hors_topic": float((sujets == -1).mean()),
        "part_du_plus_grand_topic_parmi_les_classees": float(tailles.max() / tailles.sum()),
        "topics_de_10_critiques_ou_moins": int((tailles <= 10).sum()),
        "tailles": sorted((int(t) for t in tailles), reverse=True),
    }


def reaffecter_hors_topic(embeddings: np.ndarray, sujets: np.ndarray) -> np.ndarray:
    """Rattache chaque critique −1 au centre de topic le plus proche (cosinus). Sert de contrôle."""
    topics = np.unique(sujets[sujets != -1])
    centres = np.vstack([embeddings[sujets == t].mean(axis=0) for t in topics])
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    sortie = sujets.copy()
    hors = np.flatnonzero(sujets == -1)
    sortie[hors] = topics[np.argmax(embeddings[hors] @ centres.T, axis=1)]
    return sortie


def _une_colonne_par_topic(sujets: np.ndarray, valeurs: list[int]) -> np.ndarray:
    return np.column_stack([(sujets == v).astype(float) for v in valeurs])


def mesurer_polarite() -> dict:
    """`topics_per_class` et écarts de prévalence pour BERTopic k=8, sur l'échantillon d'entraînement."""
    if CHEMIN_POLARITE.exists():
        return json.loads(CHEMIN_POLARITE.read_text(encoding="utf-8"))
    comparaison = bm.executer_comparaison()
    d = bm._donnees_echantillon()
    debut = time.perf_counter()
    bm.calculer_umap(d["embeddings"], bm.GRAINE)
    duree_umap = time.perf_counter() - debut
    natif, _ = bm.ajuster_bertopic(d["documents"], d["embeddings"], d["coordonnees"],
                                   d["matrices"]["termes"], comparaison["taille_min"])
    reduit, sujets = bm.reduire_a_k(natif, d["documents"], 8)
    cache = np.load(bm.DOSSIER_MODELES / "sujets_bertopic_k8.npy")
    y = d["echantillon"]["y"].to_numpy()
    classes = ["positif" if v == 1 else "négatif" for v in y]
    par_classe = reduit.topics_per_class(d["documents"], classes)
    noms = charger_noms("bertopic_k8", comparaison["modeles"]["bertopic_k8"]["mots"])

    valeurs = [-1, *range(8)]
    table = tp.comparer_prevalences(_une_colonne_par_topic(sujets, valeurs), y)
    reaffectes = reaffecter_hors_topic(d["embeddings"], sujets)
    controle = tp.comparer_prevalences(_une_colonne_par_topic(reaffectes, list(range(8))), y)
    lignes = []
    for rang, valeur in enumerate(valeurs):
        ligne = table.iloc[rang]
        mots = par_classe.loc[par_classe["Topic"] == valeur]
        lignes.append({
            "topic": valeur, "nom": "Hors topic (−1)" if valeur == -1 else noms[valeur]["nom"],
            "critiques": int((sujets == valeur).sum()),
            **{c: float(ligne[c]) for c in ("prevalence_positif_pct", "prevalence_negatif_pct", "ecart_points",
                                            "ic95_bas", "ic95_haut", "d_cohen")},
            "mots_par_classe": {r["Class"]: r["Words"] for _, r in mots.iterrows()},
        })
    resultat = {
        "partition_identique_au_cache": bool(np.array_equal(sujets, cache)),
        "duree_umap_s": round(duree_umap, 1),
        "topics": lignes,
        "controle_hors_topic_reaffecte": [
            {"topic": t, "nom": noms[t]["nom"], "ecart_points": float(controle.iloc[t]["ecart_points"]),
             "part_des_critiques_pct": float(100 * (reaffectes == t).mean())}
            for t in range(8)
        ],
    }
    bm.ecrire_json(CHEMIN_POLARITE, resultat)
    return resultat


def mesurer_poids() -> dict:
    """Taille sur disque des paquets ajoutés, de l'encodeur et des modèles."""
    avant = {l.split("==")[0].lower() for l in (bm.DOSSIER / "pip_freeze_avant.txt").read_text().splitlines()}
    apres = {l.split("==")[0].lower() for l in (bm.DOSSIER / "pip_freeze_apres.txt").read_text().splitlines()}
    octets = 0
    for nom in sorted(apres - avant):
        for fichier in im.distribution(nom).files or []:
            chemin = fichier.locate()
            if chemin.exists():
                octets += chemin.stat().st_size
    taille = lambda p: sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) if p.is_dir() else p.stat().st_size
    poids = {
        "paquets_ajoutes": len(apres - avant),
        "paquets_ajoutes_mo": round(octets / 1024 ** 2, 1),
        "encodeur_mo": round((bm.DOSSIER_ENCODEUR / "model.safetensors").stat().st_size / 1024 ** 2, 1),
        "bertopic_k8_sans_encodeur_mo": round(taille(bm.DOSSIER_MODELES / "bertopic_k8") / 1024 ** 2, 2),
        "nmf_k8_mo": round(taille(RACINE / "data/interim/phase4/modeles/nmf_k08_mindf10_g42.joblib") / 1024 ** 2, 2),
    }
    bm.ecrire_json(CHEMIN_POIDS, poids)
    return poids


def figure_comparaison(modeles: dict) -> None:
    ex.appliquer_style()
    panneaux = (("c_v", "Cohérence c_v"), ("diversite", "Diversité"),
                ("stabilite_jaccard", "Stabilité (Jaccard)"), ("part_hors_topic", "Critiques hors topic"))
    fig, axes = plt.subplots(1, 4, figsize=(18, 6.6), sharey=True)
    y = np.arange(len(ORDRE_FIGURE))[::-1]
    for ax, (cle, titre) in zip(axes, panneaux):
        valeurs = [modeles[n][cle] for n, _ in ORDRE_FIGURE]
        couleurs = [COULEUR_RETENU if n.startswith("nmf") else COULEUR_NEUTRE for n, _ in ORDRE_FIGURE]
        ax.barh(y, [v or 0 for v in valeurs], height=0.62, color=couleurs)
        for position, valeur in zip(y, valeurs):
            if valeur is None:
                texte = "non mesuré"
            elif cle == "part_hors_topic":
                texte = f"{_fr(100 * valeur, 1)} %"
            else:
                texte = _fr(valeur)
            ax.text((valeur or 0) + 0.02, position, texte, va="center", fontsize=16, color=ex.ENCRE)
        ax.set_xlim(0, 1.32)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(titre, loc="left", fontsize=18)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([l for _, l in ORDRE_FIGURE])
    fig.suptitle("BERTopic gagne en cohérence, mais perd en couverture et en stabilité",
                 x=0.01, ha="left", fontsize=22, fontweight="bold")
    fig.text(0.01, -0.04, f"5 000 critiques, mêmes mots possibles, même calcul de cohérence. Violet : modèle commun. {SOURCE}.",
             fontsize=16, color=ex.ENCRE_2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ex._sauver(fig, "17_bertopic_comparaison.png")


def figure_polarite(polarite: dict) -> None:
    ex.appliquer_style()
    lignes = sorted(polarite["topics"], key=lambda l: l["ecart_points"])
    fig, ax = plt.subplots(figsize=(16, 8))
    y = np.arange(len(lignes))
    ecarts = np.array([l["ecart_points"] for l in lignes])
    ax.barh(y, ecarts, height=0.64, color=[ex.COULEURS["positive"] if v > 0 else ex.COULEURS["negative"] for v in ecarts])
    bas, haut = np.array([l["ic95_bas"] for l in lignes]), np.array([l["ic95_haut"] for l in lignes])
    ax.errorbar(ecarts, y, xerr=[ecarts - bas, haut - ecarts], fmt="none", ecolor=ex.ENCRE, elinewidth=1.5, capsize=4)
    ax.axvline(0, color=ex.ENCRE_2, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l['nom']} ({format(l['critiques'], ',').replace(',', ' ')})" for l in lignes])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    marge = max(abs(bas).max(), abs(haut).max()) * 0.32
    ax.set_xlim(bas.min() - marge, haut.max() + marge)
    for position, ligne in zip(y, lignes):
        valeur = ligne["ecart_points"]
        depart = ligne["ic95_haut"] if valeur > 0 else ligne["ic95_bas"]
        ax.annotate(f"{valeur:+.1f} pt".replace(".", ",").replace("-", "−"), (depart, position),
                    xytext=(7 if valeur > 0 else -7, 0), textcoords="offset points", va="center",
                    ha="left" if valeur > 0 else "right", fontsize=16)
    ax.set_xlabel("Écart de part des critiques, positif moins négatif (points de %), IC bootstrap à 95 %")
    ax.set_title("BERTopic k=8 : peu de topics assez grands pour comparer les polarités", loc="left",
                 fontweight="bold", pad=44)
    ax.legend(handles=[Patch(color=ex.COULEURS["negative"], label="Plus présent dans les avis négatifs"),
                       Patch(color=ex.COULEURS["positive"], label="Plus présent dans les avis positifs")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    fig.text(0.01, -0.03, f"Entre parenthèses : nombre de critiques sur 5 000. {SOURCE}.", fontsize=16, color=ex.ENCRE_2)
    ex._sauver(fig, "18_bertopic_polarite.png")


def figure_phrases(croisement: dict, noms: list[dict]) -> None:
    ex.appliquer_style()
    lignes = sorted(croisement["aspects"], key=lambda a: a["critiques_positives"]["negative"]["part_pct"])
    fig, axes = plt.subplots(1, 2, figsize=(18, 9.2), sharey=True)
    y = np.arange(len(lignes))
    couleurs = {"negative": ex.COULEURS["negative"], "incertaine": COULEUR_INCERTAIN, "positive": ex.COULEURS["positive"]}
    for ax, cle, titre in ((axes[0], "critiques_positives", "Dans les critiques positives"),
                           (axes[1], "critiques_negatives", "Dans les critiques négatives")):
        gauche = np.zeros(len(lignes))
        for categorie in bp.CATEGORIES:
            parts = np.array([l[cle][categorie]["part_pct"] for l in lignes])
            ax.barh(y, parts, left=gauche, height=0.68, color=couleurs[categorie], edgecolor=ex.SURFACE, linewidth=2)
            if categorie != "incertaine":
                for position, part, depart in zip(y, parts, gauche):
                    ax.text(depart + part / 2, position, f"{part:.0f}", ha="center", va="center", fontsize=16,
                            color="#ffffff")
            gauche += parts
        ax.set_xlim(0, 100)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(titre, loc="left", fontsize=18)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{noms[l['aspect']]['nom']}{' *' if noms[l['aspect']]['flou'] else ''}" for l in lignes])
    fig.suptitle("Exploratoire : chaque aspect reçoit des phrases des deux tonalités", x=0.01, ha="left",
                 fontsize=22, fontweight="bold")
    fig.legend(handles=[Patch(color=couleurs["negative"], label="Score négatif (< 0,28)"),
                        Patch(color=couleurs["incertaine"], label="Incertain"),
                        Patch(color=couleurs["positive"], label="Score positif (> 0,72)")],
               loc="upper left", bbox_to_anchor=(0.01, 0.945), ncol=3, frameon=False)
    fig.text(0.01, -0.05, "Parts de phrases en %. Scores d'un modèle calibré sur des critiques entières, non validés sur des phrases.\n"
                          f"* Aspect flou à la lecture. 3 000 critiques. {SOURCE}.", fontsize=16, color=ex.ENCRE_2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    ex._sauver(fig, "19_phrases_aspects_sentiment.png")


def _resume_modele(m: dict) -> dict:
    return {k: m[k] for k in ("n_topics", "c_v", "c_npmi", "diversite", "part_hors_topic", "stabilite_jaccard")}


def executer() -> dict:
    """Assemble les résultats, trace les figures et écrit uniquement la clé `bertopic`."""
    grille = json.loads(bm.CHEMIN_GRILLE.read_text(encoding="utf-8"))
    premiere = json.loads((bm.DOSSIER / "grille_bertopic_regle1.json").read_text(encoding="utf-8"))
    comparaison = bm.executer_comparaison()
    modeles = comparaison["modeles"]
    polarite = mesurer_polarite()
    poids = mesurer_poids()
    diag = json.loads(bm.CHEMIN_DIAG_EMBEDDINGS.read_text(encoding="utf-8"))
    diag_phrases = json.loads(bm.CHEMIN_DIAG_PHRASES.read_text(encoding="utf-8"))
    chrono = json.loads((bm.DOSSIER / "chrono_embeddings.json").read_text(encoding="utf-8"))
    croisement = bp.croiser()
    grille_phrases = json.loads(bp.CHEMIN_GRILLE.read_text(encoding="utf-8"))
    noms_aspects = charger_noms("aspects", grille_phrases["mots"])
    clone = json.loads(bp.CHEMIN_CLONE_INFO.read_text(encoding="utf-8"))
    d_sujets = {n: np.load(bm.DOSSIER_MODELES / f"sujets_{n}.npy") for n in ("bertopic_natif", "bertopic_k8", "bertopic_k12")}

    # c_v pondérée par la taille des topics : un micro-topic pèse alors selon les critiques qu'il couvre.
    tailles = np.bincount(d_sujets["bertopic_k8"][d_sujets["bertopic_k8"] != -1])
    cv_ponderee = float(np.average(modeles["bertopic_k8"]["c_v_par_topic"], weights=tailles))

    figure_comparaison(modeles)
    figure_polarite(polarite)
    figure_phrases(croisement, noms_aspects)

    n_corpus = chrono["corpus"]
    bloc = {
        "statut": "comparaison descriptive, hors modèle gelé",
        "encodeur": {"nom": bm.NOM_ENCODEUR, "revision": bm.REVISION_ENCODEUR, "dimension": diag["dimension"],
                     "longueur_max_jetons": 256, "licence": "Apache-2.0", "materiel": "CPU, 8 fils"},
        "echantillon": {
            "critiques": diag["critiques"], "origine": "entraînement seul, 0 critique de test",
            "strates": "classe × quartile de longueur, moitié de chaque classe", "graine": bm.GRAINE,
            "raison": "Calcul complet extrapolé au-delà de 15 minutes.",
        },
        "temps": {
            "chrono_2000_critiques_s": chrono["mesures_par_fils"]["8"]["total_s"],
            "extrapolation_corpus_min": chrono["mesures_par_fils"]["8"]["extrapolation_min"],
            "extrapolation_corpus_avec_marge_2_4_min": 2.4 * chrono["mesures_par_fils"]["8"]["extrapolation_min"],
            "embeddings_5000_critiques_mesure_s": diag["duree_encodage_s"],
            "embeddings_phrases_mesure_s": diag_phrases["duree_encodage_s"],
            "umap_5000_critiques_s": polarite["duree_umap_s"],
            "hdbscan_et_representation_s": comparaison["durees"]["bertopic_ajustement_s"],
            "nmf_k8_echantillon_s": comparaison["durees"]["nmf_k8_echantillon_s"],
            "lda_k12_echantillon_s": comparaison["durees"]["lda_k12_echantillon_s"],
            "corpus_de_reference": n_corpus,
        },
        "troncature": {k: diag[k] for k in ("capacite_jetons_par_bloc", "part_critiques_depassant_la_capacite",
                                            "part_jetons_perdus_sans_decoupage", "blocs")}
                      | {"traitement": "découpage en blocs de phrases, moyenne pondérée par les jetons"},
        "pretraitement": {
            "embeddings": "HTML, adresses et notes retirés ; casse, ponctuation et mots vides conservés",
            "representation": "textes du pipeline B, vocabulaire de la phase 3",
            "nmf_lda": "pipeline B pour l'ajustement",
            "difference": "BERTopic regroupe sur le texte presque complet ; NMF et LDA sur des lemmes sans mots vides.",
        },
        "choix_hdbscan": {
            "premiere_regle": {"resultat": {k: premiere["choix"][k] for k in ("taille_min", "n_topics", "part_hors_topic", "plus_grand_topic")},
                               "decision": "rejetée : un topic absorbe 88 % des critiques classées et k=8 est impossible"},
            "regle_revisee": grille["regle"], "grille": grille["grille"], "choix": grille["choix"]["taille_min"],
            "limite": "Règle révisée après avoir vu la grille ; valeur retenue au bord bas de la grille.",
        },
        "comparaison": {n: _resume_modele(m) for n, m in modeles.items()},
        "concentration": {n: concentration(s) for n, s in d_sujets.items()},
        "bertopic_k8_c_v_ponderee_par_taille": cv_ponderee,
        "poids": poids,
        "polarite_k8": polarite,
        "conclusion": (
            f"BERTopic k=8 obtient une c_v de {_fr(modeles['bertopic_k8']['c_v'])} contre "
            f"{_fr(modeles['nmf_k8_corpus']['c_v'])} pour NMF k=8, mais laisse "
            f"{_fr(100 * modeles['bertopic_k8']['part_hors_topic'], 1)} % des critiques hors topic, "
            "concentre l'essentiel des critiques classées dans un topic et reste moins stable. "
            "NMF k=8 reste le modèle commun."
        ),
        "extension_phrases": {
            **{k: v for k, v in croisement.items()},
            "noms_aspects": [n["nom"] for n in noms_aspects],
            "aspects_flous": [n["nom"] for n in noms_aspects if n["flou"]],
            "mots_aspects": grille_phrases["mots"],
            "qualite_aspects": {k: grille_phrases[k] for k in ("c_v_entrainement_aspects", "diversite_aspects",
                                                               "part_hors_topic_aspects", "phrases_modelisees",
                                                               "phrases_sans_contenu_ecartees")},
            "taille_min_hdbscan": grille_phrases["choix"]["taille_min"],
            "modele_de_score": clone,
        },
        "figures": ["17_bertopic_comparaison.png", "18_bertopic_polarite.png", "19_phrases_aspects_sentiment.png"],
    }
    reference = json.loads((bm.DOSSIER / "reference_avant.json").read_text(encoding="utf-8"))["cles_metrics"]
    import hashlib

    metriques = json.loads(CHEMIN_METRIQUES.read_text(encoding="utf-8"))
    for cle, empreinte in reference.items():
        calcul = hashlib.sha256(json.dumps(metriques[cle], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if calcul != empreinte:
            raise RuntimeError(f"La clé `{cle}` de metrics.json a changé : écriture refusée.")
    ex.enregistrer_metrics("bertopic", bloc)
    return bloc


if __name__ == "__main__":
    resultat = executer()
    print(resultat["conclusion"])
    print(json.dumps({k: resultat[k] for k in ("temps", "poids", "concentration", "bertopic_k8_c_v_ponderee_par_taille")},
                     ensure_ascii=False, indent=1))
    print(json.dumps(resultat["polarite_k8"]["controle_hors_topic_reaffecte"], ensure_ascii=False))
    print(resultat["polarite_k8"]["partition_identique_au_cache"])

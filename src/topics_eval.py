"""Mesures de la phase 4 : choix de k, effet de min_df, stabilité, comparaison par polarité.

Les modèles sont ajustés sans étiquette sur tout le corpus dédoublonné. La comparaison par polarité
lit ensuite les étiquettes de tout le corpus, test compris. Ces résultats sont donc purement
descriptifs : ils n'alimentent ni n'orientent le classifieur de la phase 5.

Exemple d'utilisation (reprend les caches) :
    python -m src.topics_grille            # grille LDA/NMF
    python -m src.topics_eval --lecture    # fiche de lecture des topics
    python -m src.topics_eval              # mesures, figures et clé `topics`
"""
from __future__ import annotations

import argparse
import json
import re

import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import exploration as ex
from src import preprocessing as pp
from src import topics as tp
from src import topics_grille as tg
from src.data import charger_brut
from src.features import CHEMIN_TEXTES
from src.preprocessing import RACINE

CHEMIN_NOMS = RACINE / "src" / "ressources" / "topics_noms.json"
CHEMIN_EXTRAITS = tg.DOSSIER / "extraits_topics.json"
CHEMIN_LECTURE = tg.DOSSIER / "lecture_topics.md"

# Règle de choix de k, fixée avant de lire les résultats de la grille.
DIVERSITE_MIN = 0.70
TOLERANCE_CV = 0.01
REGLE_K = (
    "Parmi les k de diversité au moins égale à 0,70, prendre le plus petit k dont la cohérence c_v "
    "moyenne est à moins de 0,01 de la meilleure. La lecture humaine des topics valide ou rejette ce choix."
)
GRILLE_MIN_DF = (5, 10, 20)
GRAINES_STABILITE = (43, 44)
PART_SOUS_ECHANTILLON = 0.8
ALPHA0_SENSIBILITE = (50.0, 500.0, 5000.0)
N_MOTS_DISTINCTIFS = 20
N_CANDIDATS_EXTRAITS = 15
# Les extraits sont montrés en soutenance : on écarte injures et passages crus, sans toucher aux calculs.
MOTS_EXCLUS_EXTRAITS = re.compile(
    r"\b(retard\w*|rap(?:e|ed|es|ist|ing)|hanged|porn\w*|softcore|nud\w+|naked|tortur\w+|incest|"
    r"molest\w*|pedophil\w*|slut\w*|whore\w*|fuck\w*|shit\w*|bitch\w*|crap\w*|sex\w*|erotic\w*)\b",
    flags=re.IGNORECASE,
)
COULEURS_FAMILLES = {"lda": "#eda100", "nmf": "#6250d6"}  # mêmes teintes que BoW et TF-IDF en phase 3
LIBELLES_FAMILLES = {"lda": "LDA sur BoW", "nmf": "NMF sur TF-IDF"}
USAGE = (
    "Descriptif : topics ajustés sans étiquette sur les 49 575 critiques. La comparaison par polarité "
    "lit les étiquettes de tout le corpus, test compris : aucun résultat de cette phase ne doit "
    "alimenter ni orienter le classifieur de la phase 5."
)


# --- Choix de k ------------------------------------------------------------------------------

def construire_table(configs: list[dict], coherences: dict) -> pd.DataFrame:
    """Une ligne par configuration : cohérences moyennes, diversité, durée."""
    lignes = []
    for c in configs:
        nom = tg.nom_config(c["famille"], c["k"], c["min_df"], c["graine"])
        lignes.append({
            "famille": c["famille"], "k": c["k"], "min_df": c["min_df"],
            "c_v": float(np.mean(coherences[nom]["c_v"])),
            "c_npmi": float(np.mean(coherences[nom]["c_npmi"])),
            "diversite": float(c["diversite"]), "duree_s": c["duree_s"],
            "alertes_convergence": c["alertes_convergence"],
        })
    return pd.DataFrame(lignes).sort_values(["famille", "k"]).reset_index(drop=True)


def choisir_k(table: pd.DataFrame) -> dict:
    """Applique la règle de choix de k à une seule famille de modèles."""
    admissibles = table[table["diversite"] >= DIVERSITE_MIN]
    regle_appliquee = True
    if admissibles.empty:
        admissibles, regle_appliquee = table, False
    meilleur = admissibles["c_v"].max()
    proches = admissibles[admissibles["c_v"] >= meilleur - TOLERANCE_CV]
    ligne = proches.sort_values("k").iloc[0]
    return {
        "k": int(ligne["k"]),
        "c_v": float(ligne["c_v"]),
        "c_npmi": float(ligne["c_npmi"]),
        "diversite": float(ligne["diversite"]),
        "k_meilleure_c_v": int(admissibles.loc[admissibles["c_v"].idxmax(), "k"]),
        "meilleure_c_v": float(meilleur),
        "k_meilleure_npmi": int(admissibles.loc[admissibles["c_npmi"].idxmax(), "k"]),
        "k_admissibles": [int(v) for v in admissibles["k"]],
        "seuil_diversite_applicable": regle_appliquee,
    }


# --- Effet de min_df et stabilité ------------------------------------------------------------

def mesurer_effet_min_df(choix: dict) -> list[dict]:
    """Réajuste chaque modèle retenu avec min_df = 5, 10 et 20, à k constant."""
    taches = [(f, choix[f]["k"], m) for f in tg.FAMILLES for m in GRILLE_MIN_DF]
    configs = Parallel(n_jobs=tg.N_PROCESSUS, backend="loky")(
        delayed(tg.ajuster_config)(f, k, m) for f, k, m in taches
    )
    coherences = tg.calculer_coherences(configs)
    table = construire_table(configs, coherences)
    tailles = {(c["famille"], c["min_df"]): c["termes_vocabulaire"] for c in configs}
    table["termes_vocabulaire"] = [
        tailles[(f, m)] for f, m in zip(table["famille"], table["min_df"])
    ]
    return table.sort_values(["famille", "min_df"]).to_dict("records")


def _ajuster_sous_echantillon(famille: str, k: int, graine: int) -> dict:
    """Réajuste sur 80 % des critiques, tirées avec une autre graine, et renvoie les mots."""
    chemin = tg.DOSSIER / "stabilite" / f"{famille}_k{k:02d}_g{graine}.json"
    if chemin.exists():
        return json.loads(chemin.read_text(encoding="utf-8"))
    donnees = tp.charger_matrices_topics()
    n = donnees["bow"].shape[0]
    indices = np.sort(np.random.default_rng(graine).choice(n, int(PART_SOUS_ECHANTILLON * n), replace=False))
    if famille == "lda":
        composantes = tp.ajuster_lda(donnees["bow"][indices], donnees["termes"], k, graine=graine).get_topics()
    else:
        composantes = tp.ajuster_nmf(donnees["tfidf"][indices], k, graine=graine).components_
    resultat = {"famille": famille, "k": k, "graine": graine,
                "mots": tp.mots_principaux(composantes, donnees["termes"])}
    tg.ecrire_json(chemin, resultat)
    return resultat


def mesurer_stabilite(choix: dict, configs: dict) -> dict:
    """Compare les topics du modèle complet à ceux de deux sous-échantillons de 80 %."""
    taches = [(f, choix[f]["k"], g) for f in tg.FAMILLES for g in GRAINES_STABILITE]
    resultats = Parallel(n_jobs=tg.N_PROCESSUS, backend="loky")(
        delayed(_ajuster_sous_echantillon)(f, k, g) for f, k, g in taches
    )
    coherences = tg.calculer_coherences(
        [{**r, "min_df": 10} for r in resultats], "coherence_stabilite.json"
    )
    sortie = {}
    for famille in tg.FAMILLES:
        reference = configs[famille]["mots"]
        comparaisons = [tp.stabilite(reference, r["mots"]) for r in resultats if r["famille"] == famille]
        noms = [tg.nom_config(famille, choix[famille]["k"], 10, g) for g in GRAINES_STABILITE]
        sortie[famille] = {
            "k": choix[famille]["k"],
            "c_v_reajustements": [float(np.mean(coherences[n]["c_v"])) for n in noms],
            "npmi_reajustements": [float(np.mean(coherences[n]["c_npmi"])) for n in noms],
            "jaccard_moyen": float(np.mean([c["jaccard_moyen"] for c in comparaisons])),
            "jaccard_min": float(np.min([c["jaccard_min"] for c in comparaisons])),
            "jaccard_par_topic": [
                float(v) for v in np.mean([c["jaccard_par_topic"] for c in comparaisons], axis=0)
            ],
            "methode": "2 réajustements sur 80 % des critiques, graines 43 et 44, topics appariés (Jaccard des 10 mots)",
        }
    return sortie


# --- Parts des topics dans chaque critique ---------------------------------------------------

def charger_theta(famille: str, k: int, donnees: dict) -> np.ndarray:
    """P(topic | critique), en cache. Pour NMF, les poids sont ramenés à une somme de 1."""
    nom = tg.nom_config(famille, k)
    chemin = tg.DOSSIER / f"theta_{nom}.npy"
    if chemin.exists():
        return np.load(chemin)
    modele = joblib.load(tg.DOSSIER_MODELES / f"{nom}.joblib")
    if famille == "lda":
        theta = tp.theta_lda(modele, donnees["bow"])
    else:
        theta = tp.normaliser_lignes(modele.transform(donnees["tfidf"]))
    np.save(chemin, theta.astype(np.float32))
    return np.load(chemin)


def preparer_extraits(famille: str, config: dict, theta: np.ndarray, id_brut: np.ndarray) -> list[list[dict]]:
    """Trois critiques représentatives par topic, choisies sans regarder les étiquettes."""
    table = pd.read_parquet(CHEMIN_TEXTES, columns=["id_brut", "texte_b"]).set_index("id_brut")
    textes_b = table["texte_b"].reindex(id_brut).to_numpy()
    longueurs = np.array([len(t.split()) for t in textes_b])
    brut = charger_brut().set_index("id_brut")["review"]
    poids = None
    if famille == "nmf":  # beaucoup de parts valent 1,00 : le poids brut départage ces critiques
        modele = joblib.load(tg.DOSSIER_MODELES / f"{tg.nom_config(famille, config['k'])}.joblib")
        poids = modele.transform(tp.charger_matrices_topics()["tfidf"])
    sortie = []
    for t, mots in enumerate(config["mots"]):
        indices = tp.documents_representatifs(
            theta[:, t], longueurs, textes_b, n=N_CANDIDATS_EXTRAITS,
            departage=None if poids is None else poids[:, t],
        )
        retenus = []
        for i in indices:
            extrait = tp.extrait_autour_des_mots(pp.tronc_commun(brut.loc[id_brut[i]]), mots)
            if MOTS_EXCLUS_EXTRAITS.search(extrait):  # extrait destiné à être montré en soutenance
                continue
            retenus.append({"id_brut": int(id_brut[i]), "part_topic": float(theta[i, t]), "extrait": extrait})
            if len(retenus) == 3:
                break
        sortie.append(retenus)
    return sortie


def ecrire_fiche_lecture(choix: dict, configs: dict, donnees: dict) -> None:
    """Fiche de relecture humaine : mots et extraits de chaque topic des deux modèles retenus."""
    lignes, extraits = ["# Fiche de lecture des topics (cache, non versionné)\n"], {}
    for famille in tg.FAMILLES:
        k = choix[famille]["k"]
        theta = charger_theta(famille, k, donnees)
        extraits[famille] = preparer_extraits(famille, configs[famille], theta, donnees["id_brut"])
        lignes.append(f"\n## {famille} k={k}\n")
        for t, mots in enumerate(configs[famille]["mots"]):
            lignes.append(f"\n### Topic {t} (part moyenne {100 * theta[:, t].mean():.1f} %) : {', '.join(mots)}")
            lignes += [f"- [{e['id_brut']}, {e['part_topic']:.2f}] {e['extrait']}" for e in extraits[famille][t]]
    tg.ecrire_json(CHEMIN_EXTRAITS, extraits)
    CHEMIN_LECTURE.write_text("\n".join(lignes), encoding="utf-8")


def lire_noms(famille_attendue: dict, configs: dict) -> dict:
    """Lit les noms donnés après relecture et vérifie qu'ils correspondent bien aux topics."""
    if not CHEMIN_NOMS.exists():
        raise FileNotFoundError(f"Noms des topics absents : {CHEMIN_NOMS}. Lancer --lecture puis les écrire.")
    noms = json.loads(CHEMIN_NOMS.read_text(encoding="utf-8"))
    for famille in tg.FAMILLES:
        bloc = noms["modeles"][famille]
        if bloc["k"] != famille_attendue[famille]["k"]:
            raise ValueError(f"Noms écrits pour un autre k ({famille}).")
        for t, mots in enumerate(configs[famille]["mots"]):
            if bloc["topics"][t]["signature"] != mots[:3]:
                raise ValueError(f"Topic {t} de {famille} : les mots ont changé, relire avant de nommer.")
    return noms


# --- Comparaison par polarité ----------------------------------------------------------------

def _magnitude(d: float) -> str:
    """Repères usuels de Cohen pour lire la taille d'effet."""
    d = abs(d)
    if d < 0.2:
        return "négligeable"
    if d < 0.5:
        return "faible"
    if d < 0.8:
        return "moyen"
    return "fort"


def comparer_modele(theta: np.ndarray, y: np.ndarray, noms: list[str], natures: list[str]) -> dict:
    """Prévalences, écarts et tailles d'effet d'un modèle commun aux deux classes."""
    table = tp.comparer_prevalences(theta, y)
    table.insert(1, "nom", noms)
    table.insert(2, "nature", natures)
    table["ampleur"] = table["d_cohen"].map(_magnitude)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    # Déduit : plus petit |d| qu'un test bilatéral à 5 % déclarerait significatif avec ces effectifs.
    d_detectable = 1.96 * np.sqrt(1 / n1 + 1 / n0)
    return {
        "effectifs": {"positif": n1, "negatif": n0},
        "topics": table.sort_values("ecart_points", ascending=False).to_dict("records"),
        "topics_p_inferieure_0_001": int((table["p_valeur_welch"] < 0.001).sum()),
        "topics_effet_au_moins_faible": int((table["d_cohen"].abs() >= 0.2).sum()),
        "nombre_topics": int(len(table)),
        "d_minimal_significatif_5pct_deduit": float(d_detectable),
    }


def mesurer_mots_distinctifs(bow, termes: list[str], y: np.ndarray) -> dict:
    """Log-odds ratio avec a priori de Dirichlet, et sensibilité à la force de l'a priori."""
    comptes_pos = np.asarray(bow[y == 1].sum(axis=0)).ravel()
    comptes_neg = np.asarray(bow[y == 0].sum(axis=0)).ravel()
    resultats = {a: tp.log_odds_dirichlet(comptes_pos, comptes_neg, alpha0=a) for a in ALPHA0_SENSIBILITE}
    reference = resultats[tp.ALPHA0_LOG_ODDS].assign(terme=termes)

    def extraire(table: pd.DataFrame) -> list[dict]:
        return [
            {"terme": l.terme, "z": float(l.z), "delta": float(l.delta),
             "occurrences_positif": int(l.comptes_i), "occurrences_negatif": int(l.comptes_j)}
            for l in table.itertuples()
        ]

    haut = reference.nlargest(N_MOTS_DISTINCTIFS, "z")
    bas = reference.nsmallest(N_MOTS_DISTINCTIFS, "z")
    sensibilite = {}
    for a, table in resultats.items():
        ordre = table["z"].to_numpy()
        haut_a = set(np.argsort(-ordre)[:N_MOTS_DISTINCTIFS])
        bas_a = set(np.argsort(ordre)[:N_MOTS_DISTINCTIFS])
        sensibilite[str(int(a))] = {
            "mots_communs_positifs_sur_20": len(haut_a & set(haut.index)),
            "mots_communs_negatifs_sur_20": len(bas_a & set(bas.index)),
        }
    return {
        "reference": "Monroe, Colaresi et Quinn (2008), Political Analysis 16(4), p. 372-403, DOI 10.1093/pan/mpn018",
        "equations_verifiees": "éq. 16 (log-odds), 19 (variance), 22 (score z), 23 (a priori), p. 385-388",
        "alpha0": tp.ALPHA0_LOG_ODDS,
        "alpha_w": "alpha0 × fréquence du mot dans les deux classes réunies",
        "limite": "L'a priori vient du même corpus : les données servent deux fois (voir la note 18 de l'article).",
        "occurrences": {"positif": int(comptes_pos.sum()), "negatif": int(comptes_neg.sum())},
        "plus_positifs": extraire(haut),
        "plus_negatifs": extraire(bas),
        "sensibilite_alpha0": sensibilite,
    }


# --- Figures ---------------------------------------------------------------------------------

def figure_choix_k(table: pd.DataFrame, choix: dict) -> plt.Figure:
    """Cohérences et diversité selon k, pour LDA et NMF."""
    ex.appliquer_style()
    panneaux = [("c_v", "Cohérence c_v"), ("c_npmi", "Cohérence NPMI"), ("diversite", "Diversité")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, (colonne, titre) in zip(axes, panneaux):
        for famille in tg.FAMILLES:
            sous = table[table["famille"] == famille]
            ax.plot(sous["k"], sous[colonne], color=COULEURS_FAMILLES[famille], linewidth=2.4,
                    marker="o", markersize=8, label=LIBELLES_FAMILLES[famille])
            retenu = sous[sous["k"] == choix[famille]["k"]]
            ax.scatter(retenu["k"], retenu[colonne], s=260, facecolor="none",
                       edgecolor=COULEURS_FAMILLES[famille], linewidth=2.4, zorder=4)
        ax.set_title(titre, loc="left", fontweight="bold")
        ax.set_xlabel("Nombre de topics k")
        ax.set_xticks(tg.GRILLE_K)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
        ax.grid(axis="y")
    axes[2].axhline(DIVERSITE_MIN, color=ex.ENCRE_2, linewidth=1.2, linestyle="--")
    axes[2].annotate("seuil 0,70", (tg.GRILLE_K[0], DIVERSITE_MIN), xytext=(0, 7),
                     textcoords="offset points", fontsize=16, color=ex.ENCRE_2)
    poignees, libelles = axes[0].get_legend_handles_labels()
    fig.legend(poignees, libelles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.03, f"Cercle : k retenu (LDA : {choix['lda']['k']}, NMF : {choix['nmf']['k']}). "
                         "49 575 critiques, 10 mots par topic.",
             ha="center", fontsize=16, color=ex.ENCRE_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return ex._sauver(fig, "09_choix_nombre_topics.png")


def figure_barres_divergentes(comparaison: dict, titre: str) -> plt.Figure:
    """Graphique principal : écart de prévalence par topic, positif moins négatif."""
    ex.appliquer_style()
    lignes = sorted(comparaison["topics"], key=lambda l: l["ecart_points"])
    hauteur = max(7.5, 0.72 * len(lignes) + 2.6)
    fig, ax = plt.subplots(figsize=(15, hauteur))
    y = np.arange(len(lignes))
    ecarts = np.array([l["ecart_points"] for l in lignes])
    couleurs = [ex.COULEURS["positive"] if v > 0 else ex.COULEURS["negative"] for v in ecarts]
    ax.barh(y, ecarts, height=0.66, color=couleurs, edgecolor=ex.SURFACE, linewidth=2)
    bas = ecarts - np.array([l["ic95_bas"] for l in lignes])
    haut = np.array([l["ic95_haut"] for l in lignes]) - ecarts
    ax.errorbar(ecarts, y, xerr=[bas, haut], fmt="none", ecolor=ex.ENCRE, elinewidth=1.6, capsize=4)
    ax.axvline(0, color=ex.ENCRE_2, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([l["nom"] + (" *" if l["nature"] != "sujet" else "") for l in lignes])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    if any(l["nature"] != "sujet" for l in lignes):
        fig.text(0.01, -0.035, "* Topic fait en partie de mots de jugement : son lien avec la polarité est en partie mécanique.",
                 fontsize=16, color=ex.ENCRE_2)
    marge = max(abs(ecarts).max() * 0.72, 0.8)  # place des étiquettes de valeur
    ax.set_xlim(min(ecarts.min(), 0) - marge, max(ecarts.max(), 0) + marge)
    for position, ligne in zip(y, lignes):
        valeur = ligne["ecart_points"]
        texte = f"{valeur:+.1f} pt (d = {ligne['d_cohen']:+.2f})".replace(".", ",").replace("-", "−")
        depart = ligne["ic95_haut"] if valeur > 0 else ligne["ic95_bas"]
        ax.annotate(texte, (depart, position), xytext=(8 if valeur > 0 else -8, 0),
                    textcoords="offset points", va="center", ha="left" if valeur > 0 else "right",
                    fontsize=16, color=ex.ENCRE)
    ax.set_xlabel("Écart de prévalence moyenne, positif moins négatif (points de %), avec IC bootstrap à 95 %")
    ax.set_title(titre, loc="left", fontweight="bold", pad=52)
    ax.legend(
        handles=[Patch(color=ex.COULEURS["negative"], label="Plus présent dans les avis négatifs"),
                 Patch(color=ex.COULEURS["positive"], label="Plus présent dans les avis positifs")],
        loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2, frameon=False,
    )
    return ex._sauver(fig, "10_topics_barres_divergentes.png")


def figure_mots_distinctifs(mots: dict, n: int = 15) -> plt.Figure:
    """Mots les plus associés à chaque classe, selon le score z du log-odds."""
    ex.appliquer_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    for ax, cle, classe in ((axes[0], "plus_negatifs", "negative"), (axes[1], "plus_positifs", "positive")):
        lignes = mots[cle][:n][::-1]
        valeurs = [abs(l["z"]) for l in lignes]
        ax.barh(range(n), valeurs, height=0.68, color=ex.COULEURS[classe], edgecolor=ex.SURFACE, linewidth=2)
        ax.set_yticks(range(n))
        ax.set_yticklabels([l["terme"].replace("_", " ") for l in lignes])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x")
        ax.set_title(f"Associés aux avis {'négatifs' if classe == 'negative' else 'positifs'}",
                     loc="left", fontweight="bold")
        ax.set_xlabel("|score z| du log-odds")
    fig.text(0.5, -0.02, "Log-odds ratio avec a priori de Dirichlet (Monroe et al., 2008), vocabulaire du pipeline B. Association, pas causalité.",
             ha="center", fontsize=16, color=ex.ENCRE_2)
    fig.tight_layout()
    return ex._sauver(fig, "11_mots_distinctifs_log_odds.png")


# --- Enchaînement ----------------------------------------------------------------------------

def _preparer() -> tuple[pd.DataFrame, dict, dict, dict, list[dict]]:
    """Relit la grille en cache et applique la règle de choix de k."""
    configs_grille = [tg.ajuster_config(f, k) for k in tg.GRILLE_K for f in tg.FAMILLES]
    coherences = tg.calculer_coherences(configs_grille)
    table = construire_table(configs_grille, coherences)
    choix = {f: choisir_k(table[table["famille"] == f]) for f in tg.FAMILLES}
    retenus = {
        f: next(c for c in configs_grille if c["famille"] == f and c["k"] == choix[f]["k"])
        for f in tg.FAMILLES
    }
    return table, choix, retenus, coherences, configs_grille


def preparer_lecture() -> None:
    """Écrit la fiche de lecture des deux modèles retenus par la règle."""
    table, choix, retenus, _, _ = _preparer()
    print(table.round(4).to_string(index=False))
    print(json.dumps(choix, ensure_ascii=False, indent=1))
    ecrire_fiche_lecture(choix, retenus, tp.charger_matrices_topics())
    print("Fiche :", CHEMIN_LECTURE)


def executer() -> dict:
    """Enchaîne la phase 4 à partir des caches, puis écrit la clé `topics` de metrics.json."""
    table, choix, retenus, coherences, _ = _preparer()
    donnees = tp.charger_matrices_topics()
    noms = lire_noms(choix, retenus)
    effet_min_df = mesurer_effet_min_df(choix)
    stabilite = mesurer_stabilite(choix, retenus)
    if not CHEMIN_EXTRAITS.exists():
        ecrire_fiche_lecture(choix, retenus, donnees)
    extraits = json.loads(CHEMIN_EXTRAITS.read_text(encoding="utf-8"))

    y = tp.charger_etiquettes_descriptives(donnees["id_brut"])
    comparaisons, fiches = {}, {}
    for famille in tg.FAMILLES:
        k = choix[famille]["k"]
        theta = charger_theta(famille, k, donnees)
        bloc = noms["modeles"][famille]["topics"]
        comparaisons[famille] = comparer_modele(
            theta, y, [b["nom"] for b in bloc], [b["nature"] for b in bloc]
        )
        nom_cfg = tg.nom_config(famille, k)
        fiches[famille] = [
            {
                "topic": t, "nom": bloc[t]["nom"], "mots": retenus[famille]["mots"][t],
                "part_moyenne_pct": float(100 * theta[:, t].mean()),
                "c_v": coherences[nom_cfg]["c_v"][t], "npmi": coherences[nom_cfg]["c_npmi"][t],
                "nature": bloc[t]["nature"], "lecture": bloc[t]["lecture"],
                "id_extraits": [e["id_brut"] for e in extraits[famille][t]],
            }
            for t in range(k)
        ]
    famille_commune = noms["famille_commune"]
    mots_distinctifs = mesurer_mots_distinctifs(donnees["bow"], donnees["termes"], y)

    figure_choix_k(table, choix)
    figure_barres_divergentes(comparaisons[famille_commune], noms["titre_figure"])
    figure_mots_distinctifs(mots_distinctifs)

    bilan = {
        "usage": USAGE,
        "corpus": {"documents": int(donnees["bow"].shape[0]), "termes": len(donnees["termes"]),
                   "pipeline": "B", "min_df": 10, "max_df": 0.5},
        "configurations": {
            "lda": "gensim LdaModel sur BoW, 15 passes (cours), chunksize 2000, graine 42",
            "nmf": "scikit-learn NMF sur TF-IDF, init nndsvda, Frobenius, 400 itérations au plus, graine 42",
        },
        "grille_k": list(tg.GRILLE_K),
        "regle_choix_k": REGLE_K,
        "grille": table.to_dict("records"),
        "choix_k": choix,
        "effet_min_df": effet_min_df,
        "stabilite": stabilite,
        "modele_commun": {"famille": famille_commune, "k": choix[famille_commune]["k"],
                          "raison": noms["raison_modele_commun"]},
        "fiches_topics": fiches,
        "comparaison_polarite": comparaisons,
        "mots_distinctifs": mots_distinctifs,
        "figures": ["reports/figures/09_choix_nombre_topics.png",
                    "reports/figures/10_topics_barres_divergentes.png",
                    "reports/figures/11_mots_distinctifs_log_odds.png"],
    }
    ex.enregistrer_metrics("topics", bilan)
    return bilan


if __name__ == "__main__":
    analyseur = argparse.ArgumentParser(description="Mesures et figures de la phase 4.")
    analyseur.add_argument("--lecture", action="store_true", help="Écrit la fiche de lecture des topics.")
    arguments = analyseur.parse_args()
    if arguments.lecture:
        preparer_lecture()
    else:
        executer()

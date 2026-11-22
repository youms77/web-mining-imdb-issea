"""Topic modeling descriptif : LDA sur BoW, NMF sur TF-IDF, comparaison par polarité.

Règle d'usage : les topics sont ajustés SANS étiquette sur les 49 575 critiques dédoublonnées.
La comparaison par polarité lit ensuite les étiquettes de tout le corpus, test compris.
Rien de ce module (topics, mots distinctifs, log-odds, prévalences) ne doit donc alimenter
ni orienter le classifieur de la phase 5.

Référence du log-odds : Monroe, Colaresi et Quinn (2008), « Fightin' Words », Political Analysis
16(4), p. 372-403, équations 16, 19, 22 et 23 (vérifiées dans l'article).

Exemple d'utilisation :
    from src import topics as tp
    donnees = tp.charger_matrices_topics()
    nmf = tp.ajuster_nmf(donnees["tfidf"], k=10)
    print(tp.mots_principaux(nmf.components_, donnees["termes"])[0])
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.models import CoherenceModel, LdaModel
from scipy import sparse, stats
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF

from src.features import CHEMIN_TEXTES
from src.preprocessing import RACINE

GRAINE = 42
DOSSIER_PHASE3 = RACINE / "data" / "interim" / "phase3"

# Réglages du cours pour LDA : BoW, 15 passes, autres paramètres par défaut de gensim.
PASSES_LDA = 15
TAILLE_LOT_LDA = 2000
N_MOTS = 10
ITERATIONS_NMF = 400
N_BOOTSTRAP = 2000
ALPHA0_LOG_ODDS = 500.0  # taille d'échantillon a priori, ordre de grandeur de Monroe et al., p. 388


# --- Chargement ------------------------------------------------------------------------------

def charger_matrices_topics(dossier: str | Path = DOSSIER_PHASE3) -> dict:
    """Relit les matrices de la phase 3 (`min_df=10`). Aucune étiquette n'est chargée."""
    dossier = Path(dossier)
    termes = json.loads((dossier / "topics_vocabulaire.json").read_text(encoding="utf-8"))["termes"]
    return {
        "bow": sparse.load_npz(dossier / "topics_bow.npz").tocsr(),
        "tfidf": sparse.load_npz(dossier / "topics_tfidf.npz").tocsr(),
        "termes": list(termes),
        "id_brut": pd.read_parquet(dossier / "topics_ids.parquet")["id_brut"].to_numpy(),
    }


def charger_etiquettes_descriptives(
    id_brut: np.ndarray, chemin: str | Path = CHEMIN_TEXTES
) -> np.ndarray:
    """Renvoie 1 pour positif, 0 pour négatif, dans l'ordre de `id_brut`.

    Usage strictement descriptif : ces étiquettes couvrent aussi le test.
    """
    table = pd.read_parquet(chemin, columns=["id_brut", "sentiment"]).set_index("id_brut")
    sentiments = table["sentiment"].reindex(id_brut)
    if sentiments.isna().any():
        raise ValueError("Des identifiants n'ont pas d'étiquette.")
    return (sentiments == "positive").astype(np.int8).to_numpy()


# --- Ajustement ------------------------------------------------------------------------------

def ajuster_lda(
    bow: sparse.spmatrix,
    termes: list[str],
    k: int,
    graine: int = GRAINE,
    passes: int = PASSES_LDA,
) -> LdaModel:
    """LDA de gensim comme dans le cours, avec une graine fixe et sans évaluation intermédiaire."""
    return LdaModel(
        corpus=Sparse2Corpus(sparse.csr_matrix(bow), documents_columns=False),
        id2word=dict(enumerate(termes)),
        num_topics=k,
        passes=passes,
        chunksize=TAILLE_LOT_LDA,
        random_state=graine,
        eval_every=None,
    )


def theta_lda(
    modele: LdaModel, bow: sparse.spmatrix, graine: int = GRAINE, taille_lot: int = 2000
) -> np.ndarray:
    """Part de chaque topic dans chaque critique, P(topic | document)."""
    modele.random_state = np.random.RandomState(graine)  # l'inférence tire son point de départ
    bow = sparse.csr_matrix(bow)
    blocs = []
    for debut in range(0, bow.shape[0], taille_lot):
        lot = list(Sparse2Corpus(bow[debut:debut + taille_lot], documents_columns=False))
        gamma, _ = modele.inference(lot)
        blocs.append(gamma)
    return normaliser_lignes(np.vstack(blocs))


def ajuster_nmf(tfidf: sparse.spmatrix, k: int, graine: int = GRAINE) -> NMF:
    """NMF sur TF-IDF. L'initialisation NNDSVDa ne dépend presque pas du hasard."""
    modele = NMF(
        n_components=k, init="nndsvda", solver="cd", beta_loss="frobenius",
        max_iter=ITERATIONS_NMF, tol=1e-4, random_state=graine,
    )
    modele.fit(tfidf)
    return modele


def normaliser_lignes(matrice: np.ndarray) -> np.ndarray:
    """Ramène chaque ligne à une somme de 1. Une ligne nulle reçoit une part égale par topic."""
    matrice = np.asarray(matrice, dtype=np.float64)
    sommes = matrice.sum(axis=1, keepdims=True)
    sortie = np.divide(matrice, sommes, out=np.zeros_like(matrice), where=sommes > 0)
    sortie[sommes.ravel() <= 0] = 1.0 / matrice.shape[1]
    return sortie


# --- Qualité des topics ----------------------------------------------------------------------

def mots_principaux(composantes: np.ndarray, termes: list[str], n: int = N_MOTS) -> list[list[str]]:
    """Les `n` mots de plus fort poids de chaque topic."""
    ordre = np.argsort(-np.asarray(composantes), axis=1, kind="stable")[:, :n]
    return [[termes[i] for i in ligne] for ligne in ordre]


def diversite(topics: list[list[str]]) -> float:
    """Part de mots uniques parmi les mots principaux : 1 signifie aucun mot partagé."""
    tous = [mot for topic in topics for mot in topic]
    return len(set(tous)) / len(tous) if tous else 0.0


def coherence_par_topic(
    topics: list[list[str]],
    textes: list[list[str]],
    dictionnaire: Dictionary,
    mesure: str = "c_v",
    n: int = N_MOTS,
) -> list[float]:
    """Cohérence de chaque topic, calculée par gensim sur les textes du pipeline B."""
    modele = CoherenceModel(
        topics=topics, texts=textes, dictionary=dictionnaire, coherence=mesure, topn=n, processes=1
    )
    return [float(v) for v in modele.get_coherence_per_topic()]


def stabilite(topics_a: list[list[str]], topics_b: list[list[str]]) -> dict:
    """Apparie deux jeux de topics pour maximiser le Jaccard moyen de leurs mots principaux."""
    jaccard = np.array([
        [len(set(a) & set(b)) / len(set(a) | set(b)) for b in topics_b] for a in topics_a
    ])
    lignes, colonnes = linear_sum_assignment(-jaccard)
    valeurs = jaccard[lignes, colonnes]
    return {
        "jaccard_moyen": float(valeurs.mean()),
        "jaccard_min": float(valeurs.min()),
        "appariement": [int(c) for c in colonnes],
        "jaccard_par_topic": [float(v) for v in valeurs],
    }


# --- Comparaison par polarité ----------------------------------------------------------------

def _bootstrap_moyennes(
    theta: np.ndarray, n_tirages: int, generateur: np.random.Generator, taille_bloc: int = 100
) -> np.ndarray:
    """Moyennes par topic sur `n_tirages` rééchantillons avec remise."""
    n = theta.shape[0]
    sorties = np.empty((n_tirages, theta.shape[1]))
    for debut in range(0, n_tirages, taille_bloc):
        fin = min(debut + taille_bloc, n_tirages)
        poids = np.vstack([
            np.bincount(generateur.integers(0, n, n), minlength=n) for _ in range(fin - debut)
        ])
        sorties[debut:fin] = poids @ theta / n
    return sorties


def probabilite_superiorite(positifs: np.ndarray, negatifs: np.ndarray) -> float:
    """P(part d'une critique positive > part d'une négative), ex æquo comptés pour moitié."""
    rangs = stats.rankdata(np.concatenate([positifs, negatifs]))
    n1, n0 = len(positifs), len(negatifs)
    return float((rangs[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def comparer_prevalences(
    theta: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    graine: int = GRAINE,
) -> pd.DataFrame:
    """Prévalence par classe, écart avec IC bootstrap à 95 %, tailles d'effet et p-valeur.

    Les écarts sont en points de pourcentage (positif moins négatif). Le rééchantillonnage se
    fait séparément dans chaque classe.
    """
    theta, y = np.asarray(theta, dtype=np.float64), np.asarray(y)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("y doit contenir les deux classes, codées 0 et 1.")
    positifs, negatifs = theta[y == 1], theta[y == 0]
    generateur = np.random.default_rng(graine)
    tirages_pos = _bootstrap_moyennes(positifs, n_bootstrap, generateur)
    tirages_neg = _bootstrap_moyennes(negatifs, n_bootstrap, generateur)
    ecarts = 100 * (tirages_pos - tirages_neg)

    lignes = []
    for t in range(theta.shape[1]):
        a, b = positifs[:, t], negatifs[:, t]
        ecart_type = np.sqrt(
            ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
        )
        lignes.append({
            "topic": t,
            "prevalence_positif_pct": 100 * a.mean(),
            "prevalence_negatif_pct": 100 * b.mean(),
            "ecart_points": 100 * (a.mean() - b.mean()),
            "ic95_bas": float(np.percentile(ecarts[:, t], 2.5)),
            "ic95_haut": float(np.percentile(ecarts[:, t], 97.5)),
            "d_cohen": float((a.mean() - b.mean()) / ecart_type) if ecart_type > 0 else 0.0,
            "prob_superiorite": probabilite_superiorite(a, b),
            "p_valeur_welch": float(stats.ttest_ind(a, b, equal_var=False).pvalue),
        })
    return pd.DataFrame(lignes)


def log_odds_dirichlet(
    comptes_i: np.ndarray, comptes_j: np.ndarray, alpha0: float = ALPHA0_LOG_ODDS
) -> pd.DataFrame:
    """Log-odds ratio avec a priori de Dirichlet informatif (Monroe et al., 2008).

    a_w = alpha0 * y_w / n (éq. 23), delta (éq. 16), variance sans approximation (éq. 19),
    z = delta / écart-type (éq. 22). Un z positif indique un mot plus associé au groupe i.
    """
    y_i, y_j = np.asarray(comptes_i, dtype=np.float64), np.asarray(comptes_j, dtype=np.float64)
    if y_i.shape != y_j.shape:
        raise ValueError("Les deux vecteurs de comptes doivent avoir la même taille.")
    if alpha0 <= 0:
        raise ValueError("alpha0 doit être strictement positif.")
    total = y_i + y_j
    if (total <= 0).any():
        raise ValueError("Chaque mot doit apparaître au moins une fois.")
    a = alpha0 * total / total.sum()
    reste_i = y_i.sum() + alpha0 - y_i - a
    reste_j = y_j.sum() + alpha0 - y_j - a
    delta = np.log((y_i + a) / reste_i) - np.log((y_j + a) / reste_j)
    variance = 1 / (y_i + a) + 1 / reste_i + 1 / (y_j + a) + 1 / reste_j
    return pd.DataFrame({
        "comptes_i": y_i, "comptes_j": y_j, "delta": delta, "variance": variance,
        "z": delta / np.sqrt(variance),
    })


# --- Extraits représentatifs -----------------------------------------------------------------

def extrait_autour_des_mots(texte: str, mots: list[str], taille: int = 240) -> str:
    """Fenêtre de `taille` caractères contenant le plus de mots du topic (comparés sur 5 lettres)."""
    racines = {m.replace("_", " ")[:5].lower() for m in mots}
    jetons = texte.split()
    if not jetons:
        return ""
    positions, curseur = [], 0
    for jeton in jetons:
        positions.append(curseur)
        curseur += len(jeton) + 1
    utiles = np.array([j[:5].lower() in racines for j in jetons], dtype=int)
    meilleur, score = 0, -1
    fin = 0
    somme = 0
    for debut in range(len(jetons)):
        while fin < len(jetons) and positions[fin] + len(jetons[fin]) - positions[debut] <= taille:
            somme += utiles[fin]
            fin += 1
        if somme > score:
            meilleur, score = debut, somme
        if fin > debut:
            somme -= utiles[debut]
        else:
            fin = debut + 1
    morceau, longueur = [], 0
    for jeton in jetons[meilleur:]:
        if longueur + len(jeton) + 1 > taille and morceau:
            break
        morceau.append(jeton)
        longueur += len(jeton) + 1
    prefixe = "… " if meilleur > 0 else ""
    suffixe = " …" if meilleur + len(morceau) < len(jetons) else ""
    return prefixe + " ".join(morceau) + suffixe


def documents_representatifs(
    parts: np.ndarray, longueurs: np.ndarray, textes_b: list[str] | np.ndarray,
    n: int = 3, longueur_min: int = 40, jaccard_max: float = 0.5, candidats: int = 2000,
    part_mots_uniques_min: float = 0.5, departage: np.ndarray | None = None,
) -> list[int]:
    """Critiques où le topic pèse le plus, assez longues, sans répétition ni quasi-copie.

    `departage` classe les critiques à part égale (NMF donne souvent une part de 1,00).
    """
    if departage is None:
        ordre = np.argsort(-parts, kind="stable")
    else:
        ordre = np.lexsort((-np.asarray(departage), -np.round(parts, 3)))
    retenus, ensembles = [], []
    for i in ordre[:candidats]:
        if longueurs[i] < longueur_min:
            continue
        jetons = textes_b[i].split()
        mots = set(jetons)
        if len(mots) / len(jetons) < part_mots_uniques_min:  # texte répété en boucle
            continue
        if any(len(mots & e) / len(mots | e) > jaccard_max for e in ensembles):
            continue
        retenus.append(int(i))
        ensembles.append(mots)
        if len(retenus) == n:
            break
    return retenus

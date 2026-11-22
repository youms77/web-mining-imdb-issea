"""Dépassement du Projet 4 : thèmes des 49 575 critiques IMDB, sans découpage des textes.

Ce script tourne sur Google Colab (GPU) et ne renvoie que des résultats légers :
`resultats_colab.zip` contient les chiffres, les mots des thèmes et un numéro de thème par critique.
Ni le modèle, ni les vecteurs, ni le texte des critiques ne sont dans cette archive.

Démarche :
1. Encoder chaque critique ENTIÈRE avec BGE-M3 (limite de 8 192 jetons).
2. Comparer deux textes : complet, puis sans les mots génériques (movie, film, watch...).
3. Comparer deux réductions : ACP à 50 axes, avec ou sans le premier axe.
4. Regrouper avec HDBSCAN (thèmes libres) ou KMeans (toutes les critiques classées).
5. Décrire les thèmes par c-TF-IDF, sur les textes du pipeline B.
6. Choisir le réglage par la règle `REGLE`, fixée avant le calcul et sans utiliser les étiquettes.

Le calcul reprend là où il s'est arrêté : les étapes terminées sont gardées dans `travail/`.

Exemple d'utilisation (dans Colab) :
    !python -u /content/drive/MyDrive/projet4_colab/themes_corpus_complet.py \
        --dossier /content/drive/MyDrive/projet4_colab
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

GRAINE = 42
NOM_ENTREE = "entree_colab.parquet"
NOM_ARCHIVE = "resultats_colab.zip"

# Encodeur à une révision fixée : le résultat reste reproductible.
ENCODEUR = {
    "nom": "BAAI/bge-m3",
    "revision": "5617a9f61b028005a4858fdac845db406aefb181",
    "max_jetons": 8192,
    "pooling": "cls",
    "licence": "MIT",
}

# Mots présents dans une grande partie des critiques : ils disent « c'est un film », pas de quoi il parle.
MOTS_GENERIQUES = ("movie", "movies", "film", "films", "watch", "watched", "watching")
_GENERIQUES = re.compile(r"\b(?:movies?|films?|watch(?:ed|ing)?)\b(?:['’]s)?", flags=re.IGNORECASE)
_ESPACES = re.compile(r"\s+")
_ESPACE_PONCTUATION = re.compile(r"\s+([,.;:!?])")

TEXTES = ("complet", "sans_generiques")
REDUCTIONS = ("acp50", "acp50_sans_axe1")
GRILLE = (("hdbscan", 50), ("hdbscan", 100), ("hdbscan", 200), ("hdbscan", 400),
          ("kmeans", 8), ("kmeans", 12), ("kmeans", 16), ("kmeans", 20), ("kmeans", 30))
GRILLE_ESSAI = (("hdbscan", 10), ("hdbscan", 20), ("kmeans", 8), ("kmeans", 12))
GRAINES_STABILITE = (43, 44)
PART_SOUS_ECHANTILLON = 0.8
N_MOTS = 10
TAILLE_PARTIE = 2000
N_MESURE = 2000
N_REPRESENTANTES = 5

# Règle écrite AVANT de voir les résultats. Elle ne doit pas être assouplie après coup.
REGLE = {
    "n_topics_min": 8,
    "n_topics_max": 40,
    "part_hors_topic_max": 0.20,
    "plus_grand_topic_max": 0.35,
    "diversite_min": 0.70,
    "tolerance_c_v": 0.01,
    "texte": (
        "Parmi les réglages ayant de 8 à 40 thèmes, au plus 20 % de critiques hors thème, un plus grand "
        "thème d'au plus 35 % du corpus et une diversité d'au moins 0,70, garder ceux dont la cohérence "
        "c_v est à moins de 0,01 de la meilleure. Prendre celui qui laisse le moins de critiques hors "
        "thème, puis celui qui a le moins de thèmes. Les étiquettes de sentiment ne servent pas au choix."
    ),
}

_JOURNAL: Path | None = None


# --- Outils généraux -----------------------------------------------------------------------------

def dire(message: str) -> None:
    """Affiche un message horodaté et le garde dans le journal."""
    ligne = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] {message}"
    print(ligne, flush=True)
    if _JOURNAL is not None:
        with _JOURNAL.open("a", encoding="utf-8") as fichier:
            fichier.write(ligne + "\n")


def _json_defaut(valeur):
    if isinstance(valeur, np.integer):
        return int(valeur)
    if isinstance(valeur, np.floating):
        return None if np.isnan(valeur) else float(valeur)
    if isinstance(valeur, np.ndarray):
        return valeur.tolist()
    raise TypeError(f"Valeur non sérialisable : {type(valeur)}")


def ecrire_json(chemin: Path, contenu) -> None:
    """Écriture atomique : une coupure ne laisse jamais un fichier à moitié écrit."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_name(chemin.name + ".tmp")
    temporaire.write_text(json.dumps(contenu, ensure_ascii=False, indent=1, default=_json_defaut,
                                     allow_nan=False), encoding="utf-8")
    os.replace(temporaire, chemin)


def lire_json(chemin: Path):
    return json.loads(chemin.read_text(encoding="utf-8"))


def sauver_npy(chemin: Path, tableau: np.ndarray) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_name(chemin.stem + ".tmp.npy")
    np.save(temporaire, tableau)
    os.replace(temporaire, chemin)


class Chrono:
    """Durées de calcul, conservées d'une reprise à l'autre."""

    def __init__(self, chemin: Path):
        self.chemin = chemin
        self.valeurs = lire_json(chemin) if chemin.exists() else {}

    def ajouter(self, cle: str, secondes: float) -> None:
        self.valeurs[cle] = round(self.valeurs.get(cle, 0.0) + secondes, 1)
        ecrire_json(self.chemin, self.valeurs)


def empreinte_contenu(ids, textes_legers, textes_b) -> str:
    """Empreinte du fichier d'entrée, pour vérifier qu'il n'a pas été abîmé pendant le transfert."""
    calcul = hashlib.sha256()
    for numero, leger, b in zip(ids, textes_legers, textes_b):
        calcul.update(f"{int(numero)}\x1f{leger}\x1f{b}\x1e".encode("utf-8"))
    return calcul.hexdigest()


def retirer_generiques(texte: str) -> str:
    """Retire les mots génériques du cinéma, sans toucher au reste de la phrase."""
    sortie = _ESPACE_PONCTUATION.sub(r"\1", _ESPACES.sub(" ", _GENERIQUES.sub(" ", texte))).strip()
    return sortie or texte


def charger_entree(chemin: Path, limite: int | None):
    """Lit et contrôle le fichier d'entrée. Renvoie la table, le vocabulaire et le manifeste."""
    import pyarrow.parquet as pq

    if not chemin.exists():
        raise FileNotFoundError(f"Fichier d'entrée introuvable : {chemin}")
    table = pq.read_table(chemin)
    manifeste = json.loads(table.schema.metadata[b"projet4"].decode("utf-8"))
    donnees = table.to_pandas()
    if len(donnees) != manifeste["critiques"] or donnees["id_brut"].nunique() != len(donnees):
        raise ValueError("Le fichier d'entrée ne contient pas le nombre attendu de critiques uniques.")
    empreinte = empreinte_contenu(donnees["id_brut"], donnees["texte_leger"], donnees["texte_b"])
    if empreinte != manifeste["empreinte"]:
        raise ValueError("L'empreinte du fichier d'entrée a changé : déposer à nouveau le fichier.")
    if limite:
        donnees = (donnees.groupby("sentiment", group_keys=False)
                   .sample(n=limite // 2, random_state=GRAINE).sort_values("id_brut"))
    return donnees.reset_index(drop=True), list(manifeste["termes"]), manifeste


# --- Encodage des critiques entières -------------------------------------------------------------

def taille_lot(jetons: int) -> int:
    """Plus les textes sont longs, plus le lot est petit, pour tenir dans la mémoire du GPU."""
    for seuil, taille in ((256, 64), (512, 32), (1024, 16), (2048, 8)):
        if jetons <= seuil:
            return taille
    return 4


def decouper_lots(longueurs_triees: np.ndarray) -> list[tuple[int, int]]:
    """Forme des lots consécutifs sur des longueurs triées par ordre croissant."""
    lots, debut, n = [], 0, len(longueurs_triees)
    while debut < n:
        taille = taille_lot(int(longueurs_triees[debut]))
        while taille > 1:
            reduite = taille_lot(int(longueurs_triees[min(debut + taille, n) - 1]))
            if reduite >= taille:
                break
            taille = reduite
        fin = min(debut + taille, n)
        lots.append((debut, fin))
        debut = fin
    return lots


class Encodeur:
    """Encodeur de textes entiers, avec mise en commun et normalisation des vecteurs."""

    def __init__(self, config: dict, exiger_gpu: bool):
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.gpu = torch.cuda.is_available()
        if exiger_gpu and not self.gpu:
            raise RuntimeError("Aucun GPU détecté. Dans Colab : Exécution > Modifier le type d'exécution > GPU.")
        self.appareil = "cuda" if self.gpu else "cpu"
        self.config = config
        torch.manual_seed(GRAINE)
        options = {"revision": config["revision"]} if config.get("revision") else {}
        self.tokenizer = AutoTokenizer.from_pretrained(config["nom"], **options)
        # BGE-M3 ne publie ses poids qu'en .bin : on évite ainsi toute demande de conversion sur le Hub.
        self.modele = AutoModel.from_pretrained(
            config["nom"], use_safetensors=config.get("safetensors", False), **options
        ).to(self.appareil).eval()

    def compter_jetons(self, textes: list[str]) -> np.ndarray:
        ids = self.tokenizer(textes, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
        return np.fromiter((len(x) for x in ids), dtype=np.int32, count=len(textes))

    def _encoder_lot(self, textes: list[str]) -> np.ndarray:
        torch = self.torch
        memoire_saturee = False
        try:
            entrees = self.tokenizer(textes, padding=True, truncation=True, max_length=self.config["max_jetons"],
                                     return_tensors="pt").to(self.appareil)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.gpu):
                cachees = self.modele(**entrees).last_hidden_state
                if self.config["pooling"] == "cls":
                    vecteurs = cachees[:, 0]
                else:
                    masque = entrees["attention_mask"].unsqueeze(-1).to(cachees.dtype)
                    vecteurs = (cachees * masque).sum(1) / masque.sum(1).clamp(min=1)
                return torch.nn.functional.normalize(vecteurs.float(), dim=1).cpu().numpy().astype(np.float32)
        except torch.cuda.OutOfMemoryError:
            if len(textes) == 1:
                raise
            memoire_saturee = True
        if memoire_saturee:
            torch.cuda.empty_cache()
            milieu = len(textes) // 2
            return np.vstack([self._encoder_lot(textes[:milieu]), self._encoder_lot(textes[milieu:])])
        raise RuntimeError("État inattendu pendant l'encodage.")

    def encoder(self, textes: list[str], longueurs: np.ndarray) -> np.ndarray:
        """Encode des textes triés par longueur croissante."""
        return np.vstack([self._encoder_lot(textes[d:f]) for d, f in decouper_lots(longueurs)])

    def liberer(self) -> None:
        del self.modele
        gc.collect()
        if self.gpu:
            self.torch.cuda.empty_cache()


def mesurer_debit(encodeur: Encodeur, textes: list[str], longueurs: np.ndarray, travail: Path) -> dict:
    """Chronomètre 2 000 critiques tirées au hasard et annonce la durée prévue."""
    chemin = travail / "mesure_debit.json"
    if chemin.exists():
        return lire_json(chemin)
    tirage = np.random.default_rng(GRAINE).choice(len(textes), min(N_MESURE, len(textes)), replace=False)
    tirage = tirage[np.argsort(longueurs[tirage], kind="stable")]
    debut = time.perf_counter()
    encodeur.encoder([textes[i] for i in tirage], longueurs[tirage])
    duree = time.perf_counter() - debut
    prevision = duree * float(longueurs.sum()) / max(1.0, float(longueurs[tirage].sum()))
    mesure = {"critiques": len(tirage), "duree_s": round(duree, 1),
              "prevision_par_variante_min": round(prevision / 60, 1)}
    ecrire_json(chemin, mesure)
    dire(f"Mesure : {len(tirage)} critiques en {duree:.0f} s. Encodage prévu : "
         f"{prevision / 60:.0f} min par variante de texte, pour {len(TEXTES)} variantes.")
    return mesure


def encoder_corpus(nom: str, textes: list[str], obtenir_encodeur, travail: Path, chrono: Chrono, essai: bool):
    """Encode un corpus par parties sauvegardées. Renvoie les vecteurs et le diagnostic."""
    final = travail / f"embeddings_{nom}.npy"
    chemin_diag = travail / f"embeddings_{nom}.json"
    if final.exists() and chemin_diag.exists():
        dire(f"Vecteurs « {nom} » déjà calculés.")
        return np.load(final), lire_json(chemin_diag)
    encodeur = obtenir_encodeur()
    debut_tok = time.perf_counter()
    longueurs = encodeur.compter_jetons(textes)
    chrono.ajouter(f"comptage_jetons_{nom}_s", time.perf_counter() - debut_tok)
    maximum = encodeur.config["max_jetons"]
    tronquees = int((longueurs > maximum).sum())
    if tronquees and not essai:
        raise RuntimeError(f"{tronquees} critiques dépassent {maximum} jetons : le calcul est arrêté.")
    mesurer_debit(encodeur, textes, longueurs, travail)
    ordre = np.argsort(longueurs, kind="stable")
    dossier = travail / f"parties_{nom}"
    dossier.mkdir(parents=True, exist_ok=True)
    for depart in range(0, len(textes), TAILLE_PARTIE):
        chemin = dossier / f"p{depart:06d}.npy"
        if chemin.exists():
            continue
        positions = ordre[depart:depart + TAILLE_PARTIE]
        debut = time.perf_counter()
        vecteurs = encodeur.encoder([textes[i] for i in positions], longueurs[positions])
        sauver_npy(chemin, vecteurs)
        chrono.ajouter(f"encodage_{nom}_s", time.perf_counter() - debut)
        dire(f"Encodage « {nom} » : {min(depart + TAILLE_PARTIE, len(textes))} / {len(textes)} critiques.")
    premiere = np.load(dossier / "p000000.npy")
    matrice = np.empty((len(textes), premiere.shape[1]), dtype=np.float32)
    for depart in range(0, len(textes), TAILLE_PARTIE):
        matrice[ordre[depart:depart + TAILLE_PARTIE]] = np.load(dossier / f"p{depart:06d}.npy")
    diagnostic = {
        "critiques": len(textes), "dimension": int(matrice.shape[1]), "jetons_moyenne": float(longueurs.mean()),
        "jetons_mediane": float(np.median(longueurs)), "jetons_maximum": int(longueurs.max()),
        "limite_jetons": int(maximum), "critiques_tronquees": tronquees,
    }
    sauver_npy(final, matrice)
    ecrire_json(chemin_diag, diagnostic)
    return matrice, diagnostic


# --- Réduction, regroupement et description des thèmes -------------------------------------------

class ReductionPrecalculee:
    """Réduction déjà calculée : évite de refaire UMAP pour chaque réglage du regroupement."""

    def __init__(self, coordonnees: np.ndarray):
        self.coordonnees = coordonnees

    def fit(self, X, y=None):
        if X.shape[0] != self.coordonnees.shape[0]:
            raise ValueError("Les coordonnées réduites ne correspondent pas aux documents.")
        return self

    def transform(self, X):
        if X.shape[0] != self.coordonnees.shape[0]:
            raise ValueError("Les coordonnées réduites ne correspondent pas aux documents.")
        return self.coordonnees


def calculer_acp(vecteurs: np.ndarray, graine: int) -> tuple[np.ndarray, np.ndarray]:
    """ACP centrée à 51 axes. Le premier axe peut ensuite être écarté."""
    from sklearn.decomposition import PCA

    n_axes = min(51, vecteurs.shape[0] - 1, vecteurs.shape[1])
    acp = PCA(n_components=n_axes, svd_solver="randomized", random_state=graine)
    return acp.fit_transform(vecteurs).astype(np.float32), acp.explained_variance_ratio_


def appliquer_reduction(axes: np.ndarray, reduction: str) -> np.ndarray:
    return axes[:, :50] if reduction == "acp50" else axes[:, 1:51]


def calculer_umap(X: np.ndarray, graine: int, n_composantes: int = 5, min_dist: float = 0.0) -> np.ndarray:
    from umap import UMAP

    # compatibility_layout=True : même méthode UMAP que la phase 6 du projet.
    modele = UMAP(n_neighbors=15, n_components=n_composantes, min_dist=min_dist, metric="cosine",
                  random_state=graine, compatibility_layout=True)
    return modele.fit_transform(X).astype(np.float32)


def modele_regroupement(famille: str, valeur: int, graine: int):
    if famille == "hdbscan":
        from hdbscan import HDBSCAN

        return HDBSCAN(min_cluster_size=valeur, metric="euclidean", cluster_selection_method="eom",
                       prediction_data=False)
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=valeur, n_init=10, random_state=graine)


def ajuster_bertopic(documents: list[str], vecteurs: np.ndarray, coordonnees: np.ndarray, termes: list[str],
                     famille: str, valeur: int, graine: int = GRAINE):
    """Ajuste BERTopic. Le c-TF-IDF atténue encore les mots très fréquents dans tous les thèmes."""
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    motif = r"(?u)\S+"
    comptes = CountVectorizer(vocabulary=termes, token_pattern=motif, lowercase=False).transform(documents)
    presents = np.flatnonzero(np.asarray(comptes.sum(axis=0)).ravel() > 0)
    vocabulaire = [termes[i] for i in presents]
    modele = BERTopic(
        language=None,
        umap_model=ReductionPrecalculee(coordonnees),
        hdbscan_model=modele_regroupement(famille, valeur, graine),
        vectorizer_model=CountVectorizer(vocabulary=vocabulaire, token_pattern=motif, lowercase=False),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        top_n_words=N_MOTS, calculate_probabilities=False, verbose=False,
    )
    sujets, _ = modele.fit_transform(documents, embeddings=vecteurs)
    return modele, np.asarray(sujets, dtype=np.int16)


def mots_topics(modele) -> dict[int, list[str]]:
    """Mots principaux de chaque thème, sans le groupe −1 des critiques hors thème."""
    return {t: [mot for mot, _ in modele.get_topic(t) if mot] for t in sorted(modele.get_topics()) if t != -1}


def diversite(topics: list[list[str]]) -> float:
    tous = [mot for topic in topics for mot in topic]
    return len(set(tous)) / len(tous) if tous else 0.0


def jaccard_apparie(a: list[list[str]], b: list[list[str]]) -> float:
    """Apparie deux jeux de thèmes pour maximiser le Jaccard moyen de leurs mots."""
    from scipy.optimize import linear_sum_assignment

    matrice = np.array([[len(set(x) & set(y)) / len(set(x) | set(y)) for y in b] for x in a])
    lignes, colonnes = linear_sum_assignment(-matrice)
    return float(matrice[lignes, colonnes].mean())


def nom_reglage(texte: str, reduction: str, famille: str, valeur: int) -> str:
    return f"{texte}__{reduction}__{famille}_{valeur}"


def evaluer_reglage(nom: str, documents, vecteurs, coordonnees, termes, famille: str, valeur: int,
                    dossier: Path, chrono: Chrono) -> dict:
    """Ajuste un réglage. Un échec est noté sans arrêter le reste de la grille."""
    chemin = dossier / f"{nom}.json"
    if chemin.exists():
        return lire_json(chemin)
    debut = time.perf_counter()
    try:
        modele, sujets = ajuster_bertopic(documents, vecteurs, coordonnees, termes, famille, valeur)
        topics = mots_topics(modele)
        classes = sujets[sujets != -1]
        if not len(classes):
            raise ValueError("aucun thème trouvé")
        ligne = {
            "reglage": nom, "famille": famille, "valeur": valeur, "n_topics": len(topics),
            "part_hors_topic": float((sujets == -1).mean()),
            "plus_grand_topic": float(np.bincount(classes).max() / len(sujets)),
            "diversite": diversite(list(topics.values())),
            "mots": {str(t): m for t, m in topics.items()},
            "duree_s": round(time.perf_counter() - debut, 1),
        }
        sauver_npy(dossier / f"{nom}.npy", sujets)
    except Exception as erreur:  # une configuration impossible ne doit pas perdre des heures de calcul
        ligne = {"reglage": nom, "famille": famille, "valeur": valeur, "erreur": str(erreur)[:300]}
    ecrire_json(chemin, ligne)
    chrono.ajouter("grille_bertopic_s", time.perf_counter() - debut)
    dire(f"Réglage {nom} : " + (ligne.get("erreur") or (
        f"{ligne['n_topics']} thèmes, {ligne['part_hors_topic']:.1%} hors thème, "
        f"plus grand thème {ligne['plus_grand_topic']:.1%}")))
    return ligne


def calculer_coherences(lignes: list[dict], textes: list[list[str]], travail: Path, chrono: Chrono) -> dict:
    """c_v et NPMI de tous les réglages, avec le même calcul gensim que la phase 4 du projet."""
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel

    chemin = travail / "coherences.json"
    connues = lire_json(chemin) if chemin.exists() else {}
    manquantes = [l for l in lignes if "erreur" not in l and l["reglage"] not in connues]
    if not manquantes:
        return connues
    debut = time.perf_counter()
    tous, reperes = [], {}
    for ligne in manquantes:
        topics = list(ligne["mots"].values())
        if any(len(t) < 2 for t in topics):
            connues[ligne["reglage"]] = {"c_v": None, "c_npmi": None}
            continue
        reperes[ligne["reglage"]] = (len(tous), len(topics))
        tous.extend(topics)
    if tous:
        dictionnaire = Dictionary([sorted({m for t in tous for m in t})])
        mesures = {}
        for mesure in ("c_v", "c_npmi"):
            calcul = CoherenceModel(topics=tous, texts=textes, dictionary=dictionnaire, coherence=mesure,
                                    topn=N_MOTS, processes=1)
            mesures[mesure] = [float(v) for v in calcul.get_coherence_per_topic()]
        for nom, (debut_bloc, n) in reperes.items():
            connues[nom] = {m: mesures[m][debut_bloc:debut_bloc + n] for m in mesures}
    ecrire_json(chemin, connues)
    chrono.ajouter("coherence_s", time.perf_counter() - debut)
    return connues


def _moyenne_valide(valeurs) -> float | None:
    if not valeurs or any(v is None or not np.isfinite(v) for v in valeurs):
        return None
    return float(np.mean(valeurs))


def choisir_reglage(lignes: list[dict]) -> dict | None:
    """Applique `REGLE`. Renvoie None si aucun réglage ne la respecte : elle n'est pas assouplie."""
    admissibles = [
        l for l in lignes
        if "erreur" not in l and l.get("c_v") is not None
        and REGLE["n_topics_min"] <= l["n_topics"] <= REGLE["n_topics_max"]
        and l["part_hors_topic"] <= REGLE["part_hors_topic_max"]
        and l["plus_grand_topic"] <= REGLE["plus_grand_topic_max"]
        and l["diversite"] >= REGLE["diversite_min"]
    ]
    if not admissibles:
        return None
    meilleure = max(l["c_v"] for l in admissibles)
    proches = [l for l in admissibles if l["c_v"] >= meilleure - REGLE["tolerance_c_v"]]
    return min(proches, key=lambda l: (l["part_hors_topic"], l["n_topics"], l["reglage"]))


def mesurer_stabilite(choix: dict, texte: str, reduction: str, vecteurs, documents, termes,
                      travail: Path, chrono: Chrono) -> dict:
    """Refait tout le regroupement sur deux sous-échantillons de 80 %."""
    chemin = travail / f"stabilite_{choix['reglage']}.json"
    if chemin.exists():
        return lire_json(chemin)
    debut = time.perf_counter()
    reference = list(choix["mots"].values())
    valeurs = []
    for graine in GRAINES_STABILITE:
        tirage = np.sort(np.random.default_rng(graine).choice(
            len(documents), int(PART_SOUS_ECHANTILLON * len(documents)), replace=False))
        axes, _ = calculer_acp(vecteurs[tirage], graine)
        coordonnees = calculer_umap(appliquer_reduction(axes, reduction), graine)
        modele, _ = ajuster_bertopic([documents[i] for i in tirage], vecteurs[tirage], coordonnees, termes,
                                     choix["famille"], choix["valeur"], graine)
        valeurs.append(jaccard_apparie(reference, list(mots_topics(modele).values())))
        dire(f"Stabilité, graine {graine} : Jaccard {valeurs[-1]:.3f}")
    resultat = {"jaccard_moyen": float(np.mean(valeurs)), "par_graine": valeurs,
                "part_sous_echantillon": PART_SOUS_ECHANTILLON, "graines": list(GRAINES_STABILITE)}
    ecrire_json(chemin, resultat)
    chrono.ajouter("stabilite_s", time.perf_counter() - debut)
    return resultat


def fiches_topics(choix: dict, sujets: np.ndarray, vecteurs: np.ndarray, ids: np.ndarray, y: np.ndarray) -> list[dict]:
    """Effectifs, polarité et numéros des critiques les plus proches du centre de chaque thème."""
    fiches = []
    for rang, (topic, mots) in enumerate(choix["mots"].items()):
        numero = int(topic)
        membres = np.flatnonzero(sujets == numero)
        centre = vecteurs[membres].mean(axis=0)
        centre /= np.linalg.norm(centre)
        similarites = vecteurs[membres] @ centre
        proches = membres[np.argsort(-similarites, kind="stable")[:N_REPRESENTANTES]]
        positifs = int(y[membres].sum())
        fiches.append({
            "topic": numero, "mots": mots, "effectif": int(len(membres)),
            "part_corpus": float(len(membres) / len(sujets)), "avis_positifs": positifs,
            "avis_negatifs": int(len(membres) - positifs), "c_v": choix["c_v_par_topic"][rang],
            "ids_representatifs": [int(i) for i in ids[proches]],
        })
    return fiches


def versions() -> dict:
    sortie = {"python": platform.python_version()}
    for paquet in ("torch", "transformers", "bertopic", "umap-learn", "hdbscan", "scikit-learn", "gensim",
                   "numpy", "pandas", "pyarrow"):
        try:
            sortie[paquet] = metadata.version(paquet)
        except metadata.PackageNotFoundError:
            sortie[paquet] = "absent"
    return sortie


def materiel() -> str:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


# --- Exécution complète --------------------------------------------------------------------------

def executer(args) -> Path:
    global _JOURNAL
    dossier = Path(args.dossier)
    donnees, termes, manifeste = charger_entree(dossier / NOM_ENTREE, args.limite)
    config = dict(ENCODEUR)
    if args.encodeur:
        config |= {"nom": args.encodeur, "revision": None, "max_jetons": args.max_jetons, "pooling": args.pooling,
                   "safetensors": True, "licence": "essai local"}
    cle = hashlib.sha256(f"{config['nom']}|{config['revision']}|{len(donnees)}|{manifeste['empreinte']}".encode()).hexdigest()[:12]
    travail = dossier / "travail" / cle
    travail.mkdir(parents=True, exist_ok=True)
    _JOURNAL = travail / "journal.txt"
    chrono = Chrono(travail / "durees.json")
    dire(f"Corpus contrôlé : {len(donnees)} critiques. Encodeur : {config['nom']}.")

    ids = donnees["id_brut"].to_numpy()
    documents = donnees["texte_b"].tolist()
    textes_encodeur = {"complet": donnees["texte_leger"].tolist()}
    textes_encodeur["sans_generiques"] = [retirer_generiques(t) for t in textes_encodeur["complet"]]
    grille = GRILLE_ESSAI if args.essai else GRILLE

    instance = None
    def obtenir_encodeur():
        nonlocal instance
        if instance is None:
            instance = Encodeur(config, exiger_gpu=not args.essai)
        return instance

    diagnostics = {}
    for texte in TEXTES:
        vecteurs, diagnostics[texte] = encoder_corpus(texte, textes_encodeur[texte], obtenir_encodeur, travail,
                                                      chrono, args.essai)
        del vecteurs
        gc.collect()
    if instance is not None:
        instance.liberer()

    dossier_grille = travail / "grille"
    dossier_grille.mkdir(exist_ok=True)
    lignes = []
    variances = {}
    for texte in TEXTES:
        vecteurs = np.load(travail / f"embeddings_{texte}.npy", mmap_mode="r")
        vecteurs = np.asarray(vecteurs)
        chemin_axes, chemin_var = travail / f"acp_{texte}.npy", travail / f"acp_{texte}_variance.npy"
        if chemin_axes.exists() and chemin_var.exists():
            axes, variance = np.load(chemin_axes), np.load(chemin_var)
        else:
            debut = time.perf_counter()
            axes, variance = calculer_acp(vecteurs, GRAINE)
            sauver_npy(chemin_axes, axes)
            sauver_npy(chemin_var, variance)
            chrono.ajouter("acp_s", time.perf_counter() - debut)
        variances[texte] = [float(v) for v in variance[:5]]
        for reduction in REDUCTIONS:
            chemin_umap = travail / f"umap_{texte}_{reduction}.npy"
            if chemin_umap.exists():
                coordonnees = np.load(chemin_umap)
            else:
                dire(f"UMAP : {texte}, {reduction}...")
                debut = time.perf_counter()
                coordonnees = calculer_umap(appliquer_reduction(axes, reduction), GRAINE)
                sauver_npy(chemin_umap, coordonnees)
                chrono.ajouter("umap_s", time.perf_counter() - debut)
            for famille, valeur in grille:
                nom = nom_reglage(texte, reduction, famille, valeur)
                ligne = evaluer_reglage(nom, documents, vecteurs, coordonnees, termes, famille, valeur,
                                        dossier_grille, chrono)
                lignes.append(ligne | {"texte": texte, "reduction": reduction})
        del vecteurs, axes
        gc.collect()

    dire("Calcul de la cohérence des thèmes...")
    textes_tokenises = [t.split() for t in documents]
    coherences = calculer_coherences(lignes, textes_tokenises, travail, chrono)
    for ligne in lignes:
        mesure = coherences.get(ligne["reglage"])
        if mesure:
            ligne["c_v"] = _moyenne_valide(mesure["c_v"])
            ligne["c_npmi"] = _moyenne_valide(mesure["c_npmi"])
            ligne["c_v_par_topic"] = mesure["c_v"]

    choix = choisir_reglage(lignes)
    resume = [{k: v for k, v in l.items() if k not in ("mots", "c_v_par_topic")} for l in lignes]
    resultat = {
        "statut": "essai" if args.essai else ("complet" if choix else "aucun_reglage_admissible"),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": {"critiques": len(donnees), "empreinte_entree": manifeste["empreinte"]},
        "encodeur": {k: v for k, v in config.items() if k != "safetensors"},
        "pretraitement": {
            "texte_encodeur": manifeste["pretraitement"],
            "mots_generiques_retires_dans_la_variante": list(MOTS_GENERIQUES),
            "description_des_themes": "c-TF-IDF sur les textes du pipeline B, vocabulaire de la phase 3, fréquences réduites",
        },
        "diagnostic_encodage": diagnostics, "variance_acp_5_axes": variances,
        "regle": REGLE, "grille": resume, "choix": None, "materiel": materiel(), "versions": versions(),
    }
    attributions = None
    if choix:
        dire(f"Réglage retenu : {choix['reglage']} (c_v = {choix['c_v']:.3f}).")
        vecteurs = np.load(travail / f"embeddings_{choix['texte']}.npy")
        sujets = np.load(dossier_grille / f"{choix['reglage']}.npy")
        stabilite = mesurer_stabilite(choix, choix["texte"], choix["reduction"], vecteurs, documents, termes,
                                      travail, chrono)
        # Les étiquettes n'entrent qu'ici, une fois le réglage choisi, pour décrire les thèmes.
        y = (donnees["sentiment"] == "positive").to_numpy(dtype=np.int8)
        axes = np.load(travail / f"acp_{choix['texte']}.npy")
        chemin_carte = travail / f"carte_{choix['reglage']}.npy"
        if chemin_carte.exists():
            carte = np.load(chemin_carte)
        else:
            dire("Carte en deux dimensions...")
            debut = time.perf_counter()
            carte = calculer_umap(appliquer_reduction(axes, choix["reduction"]), GRAINE, n_composantes=2, min_dist=0.1)
            sauver_npy(chemin_carte, carte)
            chrono.ajouter("carte_2d_s", time.perf_counter() - debut)
        resultat["choix"] = {k: v for k, v in choix.items() if k not in ("mots", "c_v_par_topic")} | {
            "stabilite": stabilite,
            "correlation_axe1_sentiment": float(np.corrcoef(axes[:, 0], y)[0, 1]),
            "topics": fiches_topics(choix, sujets, vecteurs, ids, y),
            "hors_topic": {
                "effectif": int((sujets == -1).sum()),
                "avis_positifs": int(y[sujets == -1].sum()),
                "avis_negatifs": int(((sujets == -1) & (y == 0)).sum()),
            },
        }
        import pandas as pd

        attributions = pd.DataFrame({
            "id_brut": ids.astype(np.int32), "topic": sujets.astype(np.int16),
            "x": carte[:, 0].astype(np.float32), "y": carte[:, 1].astype(np.float32),
        })
    else:
        dire("Aucun réglage ne respecte la règle fixée. La grille complète est exportée sans choix.")
    resultat["durees_s"] = chrono.valeurs

    sortie = dossier / NOM_ARCHIVE
    temporaire = dossier / (NOM_ARCHIVE + ".tmp")
    chemin_json = travail / "resultats.json"
    ecrire_json(chemin_json, resultat)
    with zipfile.ZipFile(temporaire, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(chemin_json, "resultats.json")
        if attributions is not None:
            chemin_attr = travail / "attributions.parquet"
            attributions.to_parquet(chemin_attr, index=False, compression="zstd")
            archive.write(chemin_attr, "attributions.parquet")
        archive.write(_JOURNAL, "journal.txt")
    os.replace(temporaire, sortie)
    dire(f"Terminé. Archive à récupérer : {sortie} ({sortie.stat().st_size / 1e6:.1f} Mo).")
    return sortie


def lire_arguments(arguments=None):
    analyseur = argparse.ArgumentParser(description="Thèmes du corpus IMDB complet, sans découpage des critiques.")
    analyseur.add_argument("--dossier", required=True, help="Dossier contenant entree_colab.parquet")
    analyseur.add_argument("--limite", type=int, default=None, help="Essai : nombre pair de critiques")
    analyseur.add_argument("--essai", action="store_true", help="Essai technique : petite grille, CPU autorisé")
    analyseur.add_argument("--encodeur", default=None, help="Essai : autre encodeur (chemin local)")
    analyseur.add_argument("--max-jetons", type=int, default=256)
    analyseur.add_argument("--pooling", choices=("cls", "mean"), default="mean")
    return analyseur.parse_args(arguments)


if __name__ == "__main__":
    try:
        executer(lire_arguments())
    except Exception as erreur:
        dire(f"ARRÊT : {type(erreur).__name__} : {erreur}")
        raise
    sys.exit(0)

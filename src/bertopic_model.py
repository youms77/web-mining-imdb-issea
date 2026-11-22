"""Phase 6 : BERTopic comparé aux modèles du cours, puis extension exploratoire par phrase.

Règles de ce module :
- Les embeddings lisent un texte légèrement nettoyé, AVEC ses mots vides. Les mots vides ne sont
  retirés que dans la représentation des topics, qui utilise les textes du pipeline B.
- Les notes écrites (« 8/10 ») sont retirées : elles recopient presque l'étiquette.
- Les topics sont descriptifs et ajustés sans étiquette. Rien ici n'alimente le modèle gelé.
- Les critiques longues sont découpées en blocs : le modèle d'embeddings tronque au-delà de sa
  longueur maximale.

Exemple d'utilisation :
    from src import bertopic_model as bm
    print(bm.decouper_phrases("Good actors. But the story goes nowhere!"))
    modele = bm.charger_encodeur()
    vecteurs, infos = bm.encoder_critiques(["Good actors. But the story goes nowhere!"], modele)
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

from src.preprocessing import _TABLE_CARACTERES, RACINE, nettoyer_html, retirer_notes, retirer_urls

GRAINE = 42
DOSSIER = RACINE / "data" / "interim" / "phase6"
DOSSIER_ENCODEUR = DOSSIER / "encodeur"

# Modèle d'embeddings, à une révision fixée pour que le résultat reste reproductible.
NOM_ENCODEUR = "sentence-transformers/all-MiniLM-L6-v2"
REVISION_ENCODEUR = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FICHIERS_ENCODEUR = (
    "README.md", "modules.json", "config.json", "config_sentence_transformers.json",
    "sentence_bert_config.json", "1_Pooling/config.json", "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
)
TAILLE_LOT = 32

_SAUT = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_ESPACES_LIGNE = re.compile(r"[^\S\n]+")
_LIGNES = re.compile(r"\s*\n\s*")
_FIN_PHRASE = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+(?=[\"'“‘(\[]*[A-Z0-9])")
_ABREVIATION = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|St|Mt|Jr|Sr|vs|etc|No|Vol|Ed|Lt|Col|Gen|Sgt|Capt)\.|\b[A-Z]\.)[\"'”’)\]]*$"
)


# --- Nettoyage léger et découpage --------------------------------------------------------------

def nettoyer_leger(texte: object) -> str:
    """Retire HTML, adresses et notes sans toucher à la casse, aux mots vides ni à la ponctuation.

    Les sauts `<br />` deviennent des retours à la ligne : ils séparent souvent deux phrases.
    """
    if not isinstance(texte, str) or not texte:
        return ""
    texte = _SAUT.sub("\n", retirer_notes(texte))
    texte = retirer_urls(nettoyer_html(texte)).translate(_TABLE_CARACTERES)
    texte = _ESPACES_LIGNE.sub(" ", texte)
    return _LIGNES.sub("\n", texte).strip()


def decouper_phrases(texte: str) -> list[str]:
    """Découpe un texte légèrement nettoyé en phrases, sans couper après « Mr. » ou une initiale."""
    phrases: list[str] = []
    for ligne in texte.split("\n"):
        morceaux: list[str] = []
        debut = 0
        for coupure in _FIN_PHRASE.finditer(ligne):
            candidat = ligne[debut:coupure.start()] + coupure.group(0).rstrip()
            if _ABREVIATION.search(candidat):
                continue
            morceaux.append(candidat.strip())
            debut = coupure.end()
        morceaux.append(ligne[debut:].strip())
        phrases.extend(m for m in morceaux if m)
    return phrases


def _ids_jetons(phrases: list[str], tokenizer) -> list[list[int]]:
    if not phrases:
        return []
    return tokenizer(phrases, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]


def composer_blocs(longueurs: list[int], maximum: int) -> list[tuple[int, int]]:
    """Regroupe des phrases consécutives en blocs d'au plus `maximum` jetons.

    Renvoie des couples (début, fin) sur les indices de phrases. Une phrase trop longue forme un
    bloc seul : elle est découpée ensuite par `decouper_jetons`.
    """
    if maximum <= 0:
        raise ValueError("maximum doit être strictement positif.")
    blocs, debut, somme = [], 0, 0
    for i, n in enumerate(longueurs):
        if i > debut and somme + n > maximum:
            blocs.append((debut, i))
            debut, somme = i, 0
        somme += n
    if longueurs:
        blocs.append((debut, len(longueurs)))
    return blocs


def decouper_jetons(ids: list[int], maximum: int) -> list[list[int]]:
    """Coupe une suite de jetons en tranches d'au plus `maximum` jetons, sans en perdre."""
    return [ids[i:i + maximum] for i in range(0, len(ids), maximum)]


# --- Encodeur ------------------------------------------------------------------------------------

def telecharger_encodeur() -> Path:
    """Télécharge uniquement les fichiers utiles (poids en safetensors) à la révision fixée."""
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    snapshot_download(
        repo_id=NOM_ENCODEUR, revision=REVISION_ENCODEUR, allow_patterns=list(FICHIERS_ENCODEUR),
        local_dir=DOSSIER_ENCODEUR,
    )
    return DOSSIER_ENCODEUR


def charger_encodeur(n_fils: int | None = None):
    """Charge l'encodeur local, sans accès réseau et sur CPU."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    import torch
    from sentence_transformers import SentenceTransformer

    if n_fils:
        torch.set_num_threads(n_fils)
    if not (DOSSIER_ENCODEUR / "model.safetensors").exists():
        raise FileNotFoundError("Encodeur absent : lancer `telecharger_encodeur()`.")
    torch.manual_seed(GRAINE)
    return SentenceTransformer(str(DOSSIER_ENCODEUR), device="cpu", local_files_only=True)


def capacite_blocs(encodeur) -> int:
    """Jetons de texte disponibles par bloc : longueur maximale moins [CLS] et [SEP]."""
    return int(encodeur.max_seq_length) - encodeur.tokenizer.num_special_tokens_to_add(pair=False)


# --- Embeddings des critiques --------------------------------------------------------------------

def preparer_blocs(textes: list[str], encodeur) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """Découpe des critiques en blocs. Renvoie blocs, n° de critique, poids en jetons, diagnostic."""
    capacite = capacite_blocs(encodeur)
    tokenizer = encodeur.tokenizer
    blocs: list[str] = []
    proprietaire: list[int] = []
    poids: list[int] = []
    longueurs_critiques = np.zeros(len(textes), dtype=np.int32)
    for numero, texte in enumerate(textes):
        phrases = decouper_phrases(nettoyer_leger(texte)) or [" "]
        ids = _ids_jetons(phrases, tokenizer)
        longueurs = [len(x) for x in ids]
        longueurs_critiques[numero] = sum(longueurs)
        for debut, fin in composer_blocs(longueurs, capacite):
            if fin - debut == 1 and longueurs[debut] > capacite:
                for tranche in decouper_jetons(ids[debut], capacite):
                    blocs.append(tokenizer.decode(tranche, skip_special_tokens=True))
                    proprietaire.append(numero)
                    poids.append(len(tranche))
            else:
                blocs.append(" ".join(phrases[debut:fin]))
                proprietaire.append(numero)
                poids.append(max(1, sum(longueurs[debut:fin])))
    diagnostic = {
        "critiques": len(textes),
        "blocs": len(blocs),
        "capacite_jetons_par_bloc": capacite,
        "part_critiques_depassant_la_capacite": float((longueurs_critiques > capacite).mean()),
        "jetons_moyens_par_critique": float(longueurs_critiques.mean()),
        "jetons_maximum": int(longueurs_critiques.max()) if len(textes) else 0,
        "part_jetons_perdus_sans_decoupage": float(
            np.maximum(longueurs_critiques - capacite, 0).sum() / max(1, longueurs_critiques.sum())
        ),
    }
    return blocs, np.array(proprietaire, dtype=np.int64), np.array(poids, dtype=np.float32), diagnostic


def agreger_blocs(vecteurs: np.ndarray, proprietaire: np.ndarray, poids: np.ndarray, n: int) -> np.ndarray:
    """Moyenne des blocs de chaque critique, pondérée par les jetons, puis norme ramenée à 1."""
    if len(proprietaire) and (proprietaire.min() < 0 or proprietaire.max() >= n):
        raise ValueError("Numéro de critique hors limites.")
    if len(set(proprietaire.tolist())) != n:
        raise ValueError("Chaque critique doit avoir au moins un bloc.")
    sortie = np.zeros((n, vecteurs.shape[1]), dtype=np.float64)
    np.add.at(sortie, proprietaire, vecteurs * poids[:, None])
    normes = np.linalg.norm(sortie, axis=1, keepdims=True)
    return (sortie / normes).astype(np.float32)


def encoder_critiques(textes: list[str], encodeur, taille_lot: int = TAILLE_LOT) -> tuple[np.ndarray, dict]:
    """Embedding de chaque critique, avec découpage en blocs. Renvoie vecteurs et diagnostic."""
    debut = time.perf_counter()
    blocs, proprietaire, poids, diagnostic = preparer_blocs(textes, encodeur)
    milieu = time.perf_counter()
    vecteurs = encodeur.encode(
        blocs, batch_size=taille_lot, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )
    fin = time.perf_counter()
    diagnostic |= {"duree_decoupage_s": milieu - debut, "duree_encodage_s": fin - milieu}
    return agreger_blocs(vecteurs, proprietaire, poids, len(textes)), diagnostic


def ecrire_json(chemin: Path, contenu) -> None:
    """Écriture atomique d'un fichier JSON."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_name(chemin.name + ".tmp")
    temporaire.write_text(json.dumps(contenu, ensure_ascii=False, indent=1), encoding="utf-8")
    temporaire.replace(chemin)


# --- Échantillon stratifié (entraînement seul) ---------------------------------------------------

N_CRITIQUES = 5000  # 0,0718 s par critique mesurée, marge 2,4 : 14,4 min prévues, sous le seuil de 15
N_CRITIQUES_PHRASES = 3000
N_MOTS_MIN_PHRASE = 4
TAILLE_MORCEAU = 500
CHEMIN_ECHANTILLON = DOSSIER / "echantillon.parquet"
CHEMIN_EMBEDDINGS = DOSSIER / "embeddings_critiques.npy"
CHEMIN_DIAG_EMBEDDINGS = DOSSIER / "embeddings_critiques.json"
CHEMIN_PHRASES = DOSSIER / "phrases.parquet"
CHEMIN_EMBEDDINGS_PHRASES = DOSSIER / "embeddings_phrases.npy"
CHEMIN_DIAG_PHRASES = DOSSIER / "embeddings_phrases.json"


def repartir_quotas(effectifs: np.ndarray, n: int) -> np.ndarray:
    """Répartit `n` tirages proportionnellement aux effectifs (méthode des plus forts restes)."""
    effectifs = np.asarray(effectifs, dtype=np.int64)
    if n > effectifs.sum():
        raise ValueError("Échantillon plus grand que la population.")
    exacts = effectifs * n / effectifs.sum()
    quotas = np.floor(exacts).astype(np.int64)
    ordre = np.argsort(-(exacts - quotas), kind="stable")
    quotas[ordre[: int(n - quotas.sum())]] += 1
    return quotas


def tirer_strates(table, n: int, graine: int = GRAINE) -> np.ndarray:
    """Tire `n` lignes : moitié par classe, proportionnel aux quartiles de longueur dans la classe.

    `table` doit contenir les colonnes `y` (0 ou 1) et `longueur`. Renvoie des positions triées.
    """
    import pandas as pd

    if n % 2:
        raise ValueError("n doit être pair pour garder autant d'avis de chaque classe.")
    generateur = np.random.default_rng(graine)
    y = table["y"].to_numpy()
    quartiles = pd.qcut(table["longueur"], 4, labels=False, duplicates="drop").to_numpy()
    positions = []
    for classe in (0, 1):
        strates = np.unique(quartiles[y == classe])
        membres = [np.flatnonzero((y == classe) & (quartiles == q)) for q in strates]
        quotas = repartir_quotas(np.array([len(m) for m in membres]), n // 2)
        for groupe, quota in zip(membres, quotas):
            positions.append(generateur.choice(groupe, int(quota), replace=False))
    return np.sort(np.concatenate(positions))


def construire_echantillon():
    """Échantillon de critiques d'ENTRAÎNEMENT. Les étiquettes du test ne sont jamais chargées."""
    import pandas as pd

    from src import features as f

    if CHEMIN_ECHANTILLON.exists():
        return pd.read_parquet(CHEMIN_ECHANTILLON)
    train = f.charger_entrainement(("id_brut", "sentiment", "texte_b")).sort_values("id_brut").reset_index(drop=True)
    table = pd.DataFrame({
        "id_brut": train["id_brut"].to_numpy(),
        "y": (train["sentiment"] == "positive").astype(np.int8).to_numpy(),
        "longueur": train["texte_b"].str.count(" ").to_numpy() + 1,
    })
    choisies = table.iloc[tirer_strates(table, N_CRITIQUES)].reset_index(drop=True)
    sous = tirer_strates(choisies, N_CRITIQUES_PHRASES, graine=GRAINE + 1)
    choisies["pour_phrases"] = False
    choisies.loc[sous, "pour_phrases"] = True
    DOSSIER.mkdir(parents=True, exist_ok=True)
    choisies.to_parquet(CHEMIN_ECHANTILLON, index=False)
    return choisies


def _textes_bruts(ids: np.ndarray) -> list[str]:
    from src.data import charger_brut

    return charger_brut()[["id_brut", "review"]].set_index("id_brut").loc[ids, "review"].tolist()


def calculer_embeddings_critiques() -> np.ndarray:
    """Embeddings de l'échantillon, sauvegardés par morceaux pour reprendre après une coupure."""
    if CHEMIN_EMBEDDINGS.exists():
        return np.load(CHEMIN_EMBEDDINGS)
    echantillon = construire_echantillon()
    textes = _textes_bruts(echantillon["id_brut"].to_numpy())
    dossier = DOSSIER / "morceaux_critiques"
    dossier.mkdir(parents=True, exist_ok=True)
    encodeur = charger_encodeur(8)
    debut = time.perf_counter()
    parties, diagnostics = [], []
    for depart in range(0, len(textes), TAILLE_MORCEAU):
        chemin, chemin_d = dossier / f"m{depart:05d}.npy", dossier / f"m{depart:05d}.json"
        if chemin.exists() and chemin_d.exists():
            parties.append(np.load(chemin))
            diagnostics.append(json.loads(chemin_d.read_text(encoding="utf-8")))
            continue
        vecteurs, diag = encoder_critiques(textes[depart:depart + TAILLE_MORCEAU], encodeur)
        np.save(chemin, vecteurs)
        ecrire_json(chemin_d, diag)
        parties.append(vecteurs)
        diagnostics.append(diag)
        print(f"Critiques encodées : {depart + len(vecteurs)} ({time.perf_counter() - debut:.0f} s)", flush=True)
    matrice = np.vstack(parties)
    poids = np.array([d["critiques"] for d in diagnostics], dtype=float)
    ecrire_json(CHEMIN_DIAG_EMBEDDINGS, {
        "critiques": int(matrice.shape[0]), "dimension": int(matrice.shape[1]),
        "blocs": int(sum(d["blocs"] for d in diagnostics)),
        "capacite_jetons_par_bloc": diagnostics[0]["capacite_jetons_par_bloc"],
        "part_critiques_depassant_la_capacite": float(np.average(
            [d["part_critiques_depassant_la_capacite"] for d in diagnostics], weights=poids)),
        "part_jetons_perdus_sans_decoupage": float(np.average(
            [d["part_jetons_perdus_sans_decoupage"] for d in diagnostics],
            weights=poids * np.array([d["jetons_moyens_par_critique"] for d in diagnostics]))),
        "jetons_moyens_par_critique": float(np.average(
            [d["jetons_moyens_par_critique"] for d in diagnostics], weights=poids)),
        "jetons_maximum": int(max(d["jetons_maximum"] for d in diagnostics)),
        "duree_encodage_s": float(sum(d["duree_decoupage_s"] + d["duree_encodage_s"] for d in diagnostics)),
    })
    np.save(CHEMIN_EMBEDDINGS, matrice)
    return matrice


def extraire_phrases(ids: np.ndarray, textes: list[str], encodeur) -> tuple[object, dict]:
    """Phrases d'au moins `N_MOTS_MIN_PHRASE` mots ; une phrase trop longue est coupée en tranches."""
    import pandas as pd

    capacite = capacite_blocs(encodeur)
    lignes, ecartees, coupees, total = [], 0, 0, 0
    for id_brut, texte in zip(ids, textes):
        phrases = decouper_phrases(nettoyer_leger(texte))
        total += len(phrases)
        gardees = [p for p in phrases if len(p.split()) >= N_MOTS_MIN_PHRASE]
        ecartees += len(phrases) - len(gardees)
        rang = 0
        for phrase, jetons in zip(gardees, _ids_jetons(gardees, encodeur.tokenizer)):
            if len(jetons) <= capacite:
                morceaux = [phrase]
            else:
                coupees += 1
                morceaux = [encodeur.tokenizer.decode(t, skip_special_tokens=True)
                            for t in decouper_jetons(jetons, capacite)]
            for morceau in morceaux:
                lignes.append((int(id_brut), rang, morceau))
                rang += 1
    table = pd.DataFrame(lignes, columns=["id_brut", "rang", "phrase"])
    return table, {
        "critiques": int(len(ids)), "phrases_detectees": int(total),
        "phrases_ecartees_moins_de_4_mots": int(ecartees), "phrases_trop_longues_coupees": int(coupees),
        "unites_encodees": int(len(table)),
    }


def calculer_embeddings_phrases() -> np.ndarray:
    """Embeddings des phrases du sous-échantillon, sauvegardés par morceaux."""
    if CHEMIN_EMBEDDINGS_PHRASES.exists():
        return np.load(CHEMIN_EMBEDDINGS_PHRASES)
    echantillon = construire_echantillon()
    ids = echantillon.loc[echantillon["pour_phrases"], "id_brut"].to_numpy()
    encodeur = charger_encodeur(8)
    table, diag = extraire_phrases(ids, _textes_bruts(ids), encodeur)
    table.to_parquet(CHEMIN_PHRASES, index=False)
    dossier = DOSSIER / "morceaux_phrases"
    dossier.mkdir(parents=True, exist_ok=True)
    phrases = table["phrase"].tolist()
    debut = time.perf_counter()
    parties = []
    for depart in range(0, len(phrases), 4000):
        chemin = dossier / f"m{depart:06d}.npy"
        if chemin.exists():
            parties.append(np.load(chemin))
            continue
        vecteurs = encodeur.encode(phrases[depart:depart + 4000], batch_size=64, convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        np.save(chemin, vecteurs)
        parties.append(vecteurs)
        print(f"Phrases encodées : {depart + len(vecteurs)} / {len(phrases)} ({time.perf_counter() - debut:.0f} s)", flush=True)
    matrice = np.vstack(parties)
    ecrire_json(CHEMIN_DIAG_PHRASES, {**diag, "duree_encodage_s": time.perf_counter() - debut})
    np.save(CHEMIN_EMBEDDINGS_PHRASES, matrice)
    return matrice


# --- BERTopic ------------------------------------------------------------------------------------

# Première grille (15, 25, 40, 60, 100) : 15 était au bord, et dès 25 un topic absorbait 88 à 93 % des critiques.
GRILLE_TAILLE_MIN = (8, 10, 12, 15, 18, 20, 25, 40, 60, 100)
DIVERSITE_MIN = 0.70  # même seuil qu'en phase 4
TOLERANCE_CV = 0.01  # même tolérance qu'en phase 4
N_TOPICS_MIN = 12  # il faut pouvoir ramener le modèle à k = 8 (NMF) et k = 12 (LDA)
PART_MAX_PLUS_GRAND_TOPIC = 0.50  # au-delà, le « topic » regroupe presque tout le corpus
K_COMPARES = (8, 12)  # nombres de topics de NMF et de LDA retenus en phase 4
GRAINES_STABILITE = (43, 44)  # mêmes graines qu'en phase 4
PART_SOUS_ECHANTILLON = 0.8
REGLE_TAILLE_MIN = (
    "Règle révisée après l'échec de la première, qui choisissait 5 topics dont un de 88 % des critiques. "
    "Parmi les valeurs de min_cluster_size donnant au moins 12 topics, un plus grand topic d'au plus 50 % "
    "des critiques classées et une diversité d'au moins 0,70, garder celles dont la c_v moyenne, calculée "
    "sur les textes d'entraînement seuls, est à moins de 0,01 de la meilleure. Prendre celle qui laisse "
    "le moins de critiques hors topic ; en cas d'égalité, la plus grande valeur."
)
CHEMIN_GRILLE = DOSSIER / "grille_bertopic.json"
CHEMIN_COMPARAISON = DOSSIER / "comparaison.json"
DOSSIER_MODELES = DOSSIER / "modeles"


class ReductionPrecalculee:
    """Réduction de dimension déjà calculée : évite de refaire UMAP à chaque réglage de HDBSCAN."""

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


def calculer_umap(embeddings: np.ndarray, graine: int = GRAINE, chemin: Path | None = None) -> np.ndarray:
    """UMAP aux réglages par défaut de BERTopic, avec une graine fixe. Résultat mis en cache."""
    if chemin is not None and chemin.exists():
        return np.load(chemin)
    from umap import UMAP

    reducteur = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=graine)
    coordonnees = reducteur.fit_transform(embeddings).astype(np.float32)
    if chemin is not None:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        np.save(chemin, coordonnees)
    return coordonnees


def ajuster_bertopic(documents: list[str], embeddings: np.ndarray, coordonnees: np.ndarray,
                     termes: list[str], taille_min: int):
    """Ajuste BERTopic. La représentation des topics utilise les textes du pipeline B.

    Le vocabulaire est celui de LDA et NMF : mêmes mots possibles, sans mots vides.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    # Un terme absent de ces documents aurait df = 0, donc un poids c-TF-IDF infini : on le retire.
    motif = r"(?u)\S+"
    comptes = CountVectorizer(vocabulary=termes, token_pattern=motif, lowercase=False).transform(documents)
    presents = np.flatnonzero(np.asarray(comptes.sum(axis=0)).ravel() > 0)
    vocabulaire = [termes[i] for i in presents]
    # language=None : avec "english", BERTopic efface les « _ » et détruit les 871 bigrammes du vocabulaire.
    # Les embeddings étant fournis, ce paramètre ne sert pas à choisir un modèle.
    modele = BERTopic(
        language=None,
        umap_model=ReductionPrecalculee(coordonnees),
        hdbscan_model=HDBSCAN(min_cluster_size=taille_min, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=False),
        vectorizer_model=CountVectorizer(vocabulary=vocabulaire, token_pattern=motif, lowercase=False),
        top_n_words=10, calculate_probabilities=False, verbose=False,
    )
    sujets, _ = modele.fit_transform(documents, embeddings=embeddings)
    return modele, np.asarray(sujets)


def reduire_a_k(modele, documents: list[str], k: int):
    """Copie du modèle ramenée à `k` topics, sans compter le topic −1."""
    import copy

    copie = copy.deepcopy(modele)
    copie.reduce_topics(documents, nr_topics=k + copie._outliers)
    sujets = np.asarray(copie.topics_)
    trouves = len(set(sujets.tolist()) - {-1})
    if trouves != k:
        raise RuntimeError(f"Réduction à {k} topics : {trouves} obtenus.")
    return copie, sujets


def mots_topics(modele) -> list[list[str]]:
    """Mots principaux de chaque topic, topic −1 exclu, du plus grand au plus petit numéro croissant."""
    sortie = []
    for topic in sorted(t for t in modele.get_topics() if t != -1):
        sortie.append([mot for mot, _ in modele.get_topic(topic) if mot])
    return sortie


def coherences_groupees(groupes: dict[str, list[list[str]]], textes: list[list[str]]) -> dict:
    """c_v et NPMI de plusieurs jeux de topics en un passage, avec le calcul de la phase 4."""
    from gensim.corpora import Dictionary

    from src import topics as tp

    tous, reperes = [], {}
    for nom, topics in groupes.items():
        reperes[nom] = (len(tous), len(topics))
        tous.extend(topics)
    dictionnaire = Dictionary([sorted({m for t in tous for m in t})])
    mesures = {m: tp.coherence_par_topic(tous, textes, dictionnaire, m) for m in ("c_v", "c_npmi")}
    return {nom: {m: mesures[m][d:d + n] for m in mesures} for nom, (d, n) in reperes.items()}


def choisir_taille_min(lignes: list[dict]) -> dict:
    """Applique `REGLE_TAILLE_MIN` à la grille."""
    admissibles = [
        l for l in lignes
        if l["n_topics"] >= N_TOPICS_MIN and l["diversite"] >= DIVERSITE_MIN
        and l["plus_grand_topic"] <= PART_MAX_PLUS_GRAND_TOPIC
    ]
    if not admissibles:
        raise RuntimeError("Aucun réglage admissible dans la grille.")
    meilleure = max(l["c_v_entrainement"] for l in admissibles)
    proches = [l for l in admissibles if l["c_v_entrainement"] >= meilleure - TOLERANCE_CV]
    return min(proches, key=lambda l: (l["part_hors_topic"], -l["taille_min"]))


def _donnees_echantillon() -> dict:
    """Embeddings, textes du pipeline B et matrices de la phase 3, restreints à l'échantillon."""
    import pandas as pd

    from src import features as f
    from src import topics as tp

    echantillon = construire_echantillon()
    ids = echantillon["id_brut"].to_numpy()
    matrices = tp.charger_matrices_topics()
    position = pd.Series(np.arange(len(matrices["id_brut"])), index=matrices["id_brut"])
    lignes = position.loc[ids].to_numpy()
    textes = pd.read_parquet(f.CHEMIN_TEXTES, columns=["id_brut", "texte_b"]).set_index("id_brut")["texte_b"]
    return {
        "echantillon": echantillon, "ids": ids, "lignes": lignes, "matrices": matrices,
        "documents": textes.loc[ids].tolist(), "embeddings": calculer_embeddings_critiques(),
        "coordonnees": calculer_umap(calculer_embeddings_critiques(), GRAINE, DOSSIER / "umap_critiques.npy"),
    }


def executer_grille() -> dict:
    """Grille HDBSCAN. Le choix n'utilise que des textes d'entraînement et aucune étiquette."""
    if CHEMIN_GRILLE.exists():
        return json.loads(CHEMIN_GRILLE.read_text(encoding="utf-8"))
    import pandas as pd

    from src import features as f
    from src import topics as tp

    debut_umap = time.perf_counter()
    d = _donnees_echantillon()
    duree_umap = time.perf_counter() - debut_umap
    lignes, groupes = [], {}
    for taille in GRILLE_TAILLE_MIN:
        debut = time.perf_counter()
        modele, sujets = ajuster_bertopic(d["documents"], d["embeddings"], d["coordonnees"],
                                          d["matrices"]["termes"], taille)
        duree = time.perf_counter() - debut
        mots = mots_topics(modele)
        groupes[str(taille)] = mots
        lignes.append({
            "taille_min": taille, "n_topics": len(mots), "part_hors_topic": float((sujets == -1).mean()),
            "plus_grand_topic": float(pd.Series(sujets[sujets != -1]).value_counts(normalize=True).iloc[0]),
            "diversite": tp.diversite(mots), "duree_s": round(duree, 1),
        })
        print(lignes[-1], flush=True)
    train = f.charger_entrainement(("id_brut", "texte_b"))
    reference = [t.split() for t in train["texte_b"]]
    mesures = coherences_groupees(groupes, reference)
    for ligne in lignes:
        valeurs = mesures[str(ligne["taille_min"])]
        ligne["c_v_entrainement"] = float(np.mean(valeurs["c_v"]))
        ligne["c_npmi_entrainement"] = float(np.mean(valeurs["c_npmi"]))
    choix = choisir_taille_min(lignes)
    resultat = {"regle": REGLE_TAILLE_MIN, "grille": lignes, "choix": choix, "duree_umap_s": round(duree_umap, 1)}
    ecrire_json(CHEMIN_GRILLE, resultat)
    return resultat


def _stabilite_cours(d: dict, famille: str, k: int, reference: list[list[str]]) -> dict:
    from src import topics as tp

    valeurs = []
    for graine in GRAINES_STABILITE:
        tirage = np.random.default_rng(graine).choice(
            len(d["ids"]), int(PART_SOUS_ECHANTILLON * len(d["ids"])), replace=False)
        lignes = d["lignes"][np.sort(tirage)]
        if famille == "nmf":
            modele = tp.ajuster_nmf(d["matrices"]["tfidf"][lignes], k, graine=graine)
            composantes = modele.components_
        else:
            composantes = tp.ajuster_lda(d["matrices"]["bow"][lignes], d["matrices"]["termes"], k, graine=graine).get_topics()
        valeurs.append(tp.stabilite(reference, tp.mots_principaux(composantes, d["matrices"]["termes"]))["jaccard_moyen"])
    return {"jaccard_moyen": float(np.mean(valeurs)), "par_graine": valeurs}


def _stabilite_bertopic(d: dict, taille: int, k: int, reference: list[list[str]]) -> dict:
    from src import topics as tp

    valeurs = []
    for graine in GRAINES_STABILITE:
        tirage = np.sort(np.random.default_rng(graine).choice(
            len(d["ids"]), int(PART_SOUS_ECHANTILLON * len(d["ids"])), replace=False))
        documents = [d["documents"][i] for i in tirage]
        coordonnees = calculer_umap(d["embeddings"][tirage], graine)
        modele, _ = ajuster_bertopic(documents, d["embeddings"][tirage], coordonnees, d["matrices"]["termes"], taille)
        reduit, _ = reduire_a_k(modele, documents, k)
        valeurs.append(tp.stabilite(reference, mots_topics(reduit))["jaccard_moyen"])
    return {"jaccard_moyen": float(np.mean(valeurs)), "par_graine": valeurs}


def executer_comparaison() -> dict:
    """Compare BERTopic, NMF et LDA avec les mêmes mesures et le même corpus de référence."""
    if CHEMIN_COMPARAISON.exists():
        return json.loads(CHEMIN_COMPARAISON.read_text(encoding="utf-8"))
    from src import topics as tp
    from src import topics_grille as tg

    grille = executer_grille()
    taille = grille["choix"]["taille_min"]
    d = _donnees_echantillon()
    termes = d["matrices"]["termes"]

    debut = time.perf_counter()
    natif, sujets_natif = ajuster_bertopic(d["documents"], d["embeddings"], d["coordonnees"], termes, taille)
    duree_natif = time.perf_counter() - debut
    variantes = {"bertopic_natif": (natif, sujets_natif)}
    for k in K_COMPARES:
        variantes[f"bertopic_k{k}"] = reduire_a_k(natif, d["documents"], k)

    groupes = {nom: mots_topics(m) for nom, (m, _) in variantes.items()}
    durees = {"bertopic_ajustement_s": duree_natif, "umap_s": grille["duree_umap_s"]}
    for nom, famille, k in (("nmf_k8", "nmf", 8), ("lda_k12", "lda", 12)):
        cfg = json.loads((tg.DOSSIER_MODELES / f"{tg.nom_config(famille, k)}.json").read_text(encoding="utf-8"))
        groupes[f"{nom}_corpus"] = cfg["mots"]
        durees[f"{nom}_corpus_s"] = cfg["duree_s"]
        debut = time.perf_counter()
        if famille == "nmf":
            composantes = tp.ajuster_nmf(d["matrices"]["tfidf"][d["lignes"]], k).components_
        else:
            composantes = tp.ajuster_lda(d["matrices"]["bow"][d["lignes"]], termes, k).get_topics()
        durees[f"{nom}_echantillon_s"] = time.perf_counter() - debut
        groupes[f"{nom}_echantillon"] = tp.mots_principaux(composantes, termes)
        print(nom, "ajusté sur l'échantillon", flush=True)

    mesures = coherences_groupees(groupes, tg.charger_textes_tokenises())
    stabilites = {
        "nmf_k8_echantillon": _stabilite_cours(d, "nmf", 8, groupes["nmf_k8_echantillon"]),
        "lda_k12_echantillon": _stabilite_cours(d, "lda", 12, groupes["lda_k12_echantillon"]),
        "bertopic_k8": _stabilite_bertopic(d, taille, 8, groupes["bertopic_k8"]),
        "bertopic_k12": _stabilite_bertopic(d, taille, 12, groupes["bertopic_k12"]),
    }
    lignes = {}
    for nom, topics in groupes.items():
        lignes[nom] = {
            "n_topics": len(topics), "c_v": float(np.mean(mesures[nom]["c_v"])),
            "c_npmi": float(np.mean(mesures[nom]["c_npmi"])), "diversite": tp.diversite(topics),
            "c_v_par_topic": mesures[nom]["c_v"], "mots": topics,
            "part_hors_topic": float((variantes[nom][1] == -1).mean()) if nom in variantes else 0.0,
            "stabilite_jaccard": stabilites.get(nom, {}).get("jaccard_moyen"),
        }
    DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
    for nom, (_, sujets) in variantes.items():
        np.save(DOSSIER_MODELES / f"sujets_{nom}.npy", sujets)
    variantes["bertopic_k8"][0].save(DOSSIER_MODELES / "bertopic_k8", serialization="safetensors",
                                     save_ctfidf=True, save_embedding_model=False)
    resultat = {"taille_min": taille, "modeles": lignes, "durees": durees, "stabilites": stabilites}
    ecrire_json(CHEMIN_COMPARAISON, resultat)
    return resultat


if __name__ == "__main__":
    import argparse

    analyseur = argparse.ArgumentParser(description="Phase 6 : embeddings, grille et comparaison.")
    analyseur.add_argument("--etape", choices=["telecharger", "embeddings", "grille", "comparer"], required=True)
    etape = analyseur.parse_args().etape
    if etape == "telecharger":
        print(telecharger_encodeur())
    elif etape == "embeddings":
        print(calculer_embeddings_critiques().shape, calculer_embeddings_phrases().shape)
    elif etape == "grille":
        print(json.dumps(executer_grille(), ensure_ascii=False, indent=1))
    else:
        print(json.dumps({k: {a: b for a, b in v.items() if a != "mots"}
                          for k, v in executer_comparaison()["modeles"].items()}, ensure_ascii=False, indent=1))

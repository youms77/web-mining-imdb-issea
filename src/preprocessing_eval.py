"""Mesures de la phase 2 : choix du prétraitement, fondés sur l'entraînement seul.

Le module produit les ressources du pipeline B, les textes en cache et les chiffres de la clé
`pretraitement` de reports/metrics.json. L'application n'a pas besoin de ce module.

Exemple d'utilisation (reprend les caches existants) :
    python -m src.preprocessing_eval
Pour tout recalculer :
    python -m src.preprocessing_eval --recalculer
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import math
import re
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src import preprocessing as pp
from src.data import charger_brut, dedoublonner, separer_entrainement_test
from src.exploration import enregistrer_metrics

GRAINE = 42
N_PROCESSUS = 6
DOSSIER_CACHE = pp.RACINE / "data" / "interim" / "phase2"
CHEMIN_TEXTES = pp.RACINE / "data" / "interim" / "textes_pretraites.parquet"

# Règles fixées avant de regarder les résultats de la validation croisée.
SEUIL_GAIN_MINIMAL = 0.0005  # 0,05 point d'exactitude
SEUIL_DF_DOMAINE = 0.20
MOTS_THEMATIQUES_GARDES = ("actor", "character", "life", "love", "plot", "scene", "story")
FORMES_SANS_APOSTROPHE = ("im", "ive", "thats", "theres", "theyre", "whats", "youre")
BIGRAMME_MIN_OCCURRENCES = 50
BIGRAMME_MIN_NPMI = 0.5
SEPARATEUR = "|"

VARIANTES_A = {
    "reference": {},
    "ponctuation": {"ponctuation": True},
    "lemmatisation": {"normalisation": "lemme"},
    "racinisation": {"normalisation": "racine"},
    "sans_contractions": {"contractions": False},
    "mots_vides_retires": {"_mots_vides": True},
    "mots_vides_sauf_negations": {"_mots_vides": True, "_garder_negations": True},
}
NEGATIONS = frozenset({"not", "no", "nor", "never"})


# --- Calcul parallèle -----------------------------------------------------------------------

def _mots_vides_anglais() -> frozenset[str]:
    pp._nltk_pret()
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english"))


def _variante_a(texte: str, reglages: dict, vides: frozenset[str] | None) -> str:
    if reglages.get("_mots_vides"):
        return " ".join(m for m in pp.tokeniser_a(pp.tronc_commun(texte)) if m not in vides)
    return pp.pipeline_a(texte, **reglages)


def _lot_variante(lot: list[str], reglages: dict) -> list[str]:
    vides = _mots_vides_anglais() if reglages.get("_mots_vides") else None
    if reglages.get("_garder_negations"):
        vides = vides - NEGATIONS
    return [_variante_a(t, reglages, vides) for t in lot]


def _sequence_b(texte: str, vides: frozenset[str]) -> str:
    """Lemmes filtrés par les seuls mots vides anglais ; « | » marque une coupure."""
    formes = pp.tokeniser_b(pp.tronc_commun(texte))
    if not formes:
        return ""
    lemmes = pp.lemmatiser(formes)
    return " ".join(m if m is not None else SEPARATEUR for m in pp.filtrer_b(formes, lemmes, vides))


def _lot_sequences(lot: list[str], vides: frozenset[str]) -> list[str]:
    return [_sequence_b(t, vides) for t in lot]


def _en_parallele(fonction, textes: list[str], *arguments, taille_lot: int = 500) -> list[str]:
    lots = [textes[i : i + taille_lot] for i in range(0, len(textes), taille_lot)]
    resultats = Parallel(n_jobs=N_PROCESSUS)(delayed(fonction)(lot, *arguments) for lot in lots)
    return [t for lot in resultats for t in lot]


# --- Mesures du nettoyage ---------------------------------------------------------------------

def mesurer_nettoyage(corpus: pd.DataFrame) -> dict:
    """Compare le motif strict de balises à un motif large, et compte ce que le tronc corrige."""
    large = re.compile(r"<[^>]+>")
    textes = corpus["review"].tolist()
    avec_br, reste_br, perte_large, critiques_perte = 0, 0, 0, 0
    negations_sans_apostrophe, formes_cannot = 0, 0
    formes = [f for f in pp._NEGATIONS_SANS_APOSTROPHE if f != "cannot"]
    motif_sans = re.compile(r"\b(?:" + "|".join(formes) + r")\b", re.IGNORECASE)
    motif_cannot = re.compile(r"\bcannot\b", re.IGNORECASE)
    entites_divergentes = 0
    for texte in textes:
        avec_br += bool(re.search(r"<br\s*/?>", texte, re.IGNORECASE))
        propre = pp.tronc_commun(texte)
        reste_br += bool(re.search(r"<br\s*/?>", propre))  # le mot « br » peut exister (abréviation)
        strict = pp._BALISE.sub(" ", texte)
        retire_en_plus = len(strict) - len(large.sub(" ", strict))
        if retire_en_plus > 0:
            perte_large += retire_en_plus
            critiques_perte += 1
        negations_sans_apostrophe += len(motif_sans.findall(texte))
        formes_cannot += len(motif_cannot.findall(texte))
        entites_divergentes += html.unescape(texte) != pp._ENTITE.sub(pp._decoder_entite, texte)
    c1 = sum(any("\x80" <= c <= "\x9f" for c in t) for t in textes)
    return {
        "critiques": len(textes),
        "critiques_avec_br_avant": avec_br,
        "critiques_avec_br_apres": reste_br,
        "motif_large_critiques_avec_texte_perdu": critiques_perte,
        "motif_large_caracteres_perdus": perte_large,
        "negations_sans_apostrophe_developpees": negations_sans_apostrophe,
        "formes_cannot_developpees": formes_cannot,
        "critiques_ou_unescape_seul_differerait": entites_divergentes,
        "critiques_avec_caracteres_windows_1252": c1,
    }


def compter_negations(textes_bruts: list[str], textes_a: list[str]) -> dict:
    """Compte les jetons « not » avant et après le pipeline A (texte découpé naïvement)."""
    avant = sum(len(re.findall(r"\bnot\b", t.lower())) for t in textes_bruts)
    apres = sum(t.split().count("not") for t in textes_a)
    contractees = sum(len(re.findall(r"\b\w+(?:n't|n[´`’]t)\b", t.lower())) for t in textes_bruts)
    return {"not_dans_le_brut": avant, "formes_nt_dans_le_brut": contractees, "not_apres_pipeline_a": apres}


# --- Pipeline B : ressources -------------------------------------------------------------------

def _frequence_documentaire(sequences: list[str]) -> collections.Counter:
    df: collections.Counter = collections.Counter()
    for s in sequences:
        df.update(set(s.split()) - {SEPARATEUR})
    return df


def choisir_mots_domaine(sequences: list[str]) -> tuple[list[str], dict]:
    """Mots présents dans au moins 20 % des critiques, sauf ceux qui nomment un thème de film."""
    df = _frequence_documentaire(sequences)
    n = len(sequences)
    au_dessus = {m: c for m, c in df.items() if c / n >= SEUIL_DF_DOMAINE}
    garder = set(MOTS_THEMATIQUES_GARDES)
    retenus = sorted(m for m in au_dessus if m not in garder)
    table = {m: round(100 * c / n, 2) for m, c in sorted(au_dessus.items(), key=lambda x: -x[1])}
    return retenus, {
        "seuil_frequence_documentaire": SEUIL_DF_DOMAINE,
        "mots_au_dessus_du_seuil": len(au_dessus),
        "mots_thematiques_gardes": sorted(garder & set(au_dessus)),
        "mots_retires": len(retenus),
        "frequence_documentaire_pct": table,
        "mots_au_dessus_de_50_pct_regle_du_cours": sorted(m for m, c in df.items() if c / n > 0.5),
    }


def choisir_bigrammes(
    sequences: list[str], mots_domaine: set[str], min_occurrences: int = BIGRAMME_MIN_OCCURRENCES
) -> tuple[list[str], dict]:
    """Expressions de deux mots voisins retenues par l'information mutuelle normalisée (NPMI).

    NPMI(x, y) = ln(p(x, y) / (p(x à gauche) p(y à droite))) / -ln p(x, y), compris entre -1 et 1.
    """
    gauche: collections.Counter = collections.Counter()
    droite: collections.Counter = collections.Counter()
    bi: collections.Counter = collections.Counter()
    for s in sequences:
        precedent = None
        for mot in s.split():
            if mot == SEPARATEUR or mot in mots_domaine:
                precedent = None
                continue
            if precedent is not None:
                bi[(precedent, mot)] += 1
                gauche[precedent] += 1
                droite[mot] += 1
            precedent = mot
    total = sum(bi.values())
    scores = {}
    for (x, y), n_xy in bi.items():
        if n_xy < min_occurrences or x == y:
            continue
        p_xy = n_xy / total
        npmi = math.log(p_xy / ((gauche[x] / total) * (droite[y] / total))) / -math.log(p_xy)
        if npmi >= BIGRAMME_MIN_NPMI:
            scores[(x, y)] = (npmi, n_xy)
    classes = sorted(scores.items(), key=lambda e: -e[1][1])
    retenus = sorted(f"{x}_{y}" for (x, y) in scores)
    return retenus, {
        "occurrences_minimales": min_occurrences,
        "npmi_minimal": BIGRAMME_MIN_NPMI,
        "expressions_retenues": len(retenus),
        "plus_frequentes": [
            {"expression": f"{x}_{y}", "occurrences": n, "npmi": round(s, 3)} for (x, y), (s, n) in classes[:25]
        ],
    }


def appliquer_ressources(sequence: str, mots_domaine: frozenset[str], bigrammes: frozenset) -> str:
    """Passe d'une séquence en cache au texte final du pipeline B."""
    mots = [None if (m == SEPARATEUR or m in mots_domaine) else m for m in sequence.split()]
    return " ".join(pp.fusionner_bigrammes(mots, bigrammes))


# --- Lemmatisation contre racinisation ----------------------------------------------------------

def comparer_lemme_racine(references: list[str], lemmes: list[str], racines: list[str]) -> dict:
    """Taille du vocabulaire et part de formes modifiées qui restent de vrais mots (WordNet)."""
    pp._nltk_pret()
    from nltk.corpus import wordnet

    memo: dict[str, bool] = {}

    def est_mot(mot: str) -> bool:
        if mot not in memo:
            memo[mot] = bool(wordnet.synsets(mot))
        return memo[mot]

    resultat: dict = {}
    vocab = {"origine": set(), "lemmatisation": set(), "racinisation": set()}
    modifies = {"lemmatisation": collections.Counter(), "racinisation": collections.Counter()}
    for ref, lem, rac in zip(references, lemmes, racines):
        r, l, s = ref.split(), lem.split(), rac.split()
        vocab["origine"].update(m for m in r if m.isalpha())
        vocab["lemmatisation"].update(m for m in l if m.isalpha())
        vocab["racinisation"].update(m for m in s if m.isalpha())
        for a, b in zip(r, l):
            if a != b and a.isalpha():
                modifies["lemmatisation"][b] += 1
        for a, b in zip(r, s):
            if a != b and a.isalpha():
                modifies["racinisation"][b] += 1
    taille_origine = len(vocab["origine"])
    resultat["vocabulaire_origine"] = taille_origine
    for nom in ("lemmatisation", "racinisation"):
        compte = modifies[nom]
        total = sum(compte.values())
        vrais = sum(c for m, c in compte.items() if est_mot(m))
        resultat[nom] = {
            "vocabulaire": len(vocab[nom]),
            "reduction_vocabulaire_pct": round(100 * (1 - len(vocab[nom]) / taille_origine), 2),
            "occurrences_modifiees": total,
            "occurrences_modifiees_vrais_mots_pct": round(100 * vrais / total, 2),
        }
    return resultat


def exemples_lemme_racine(mots: tuple[str, ...] = ("movies", "stories", "acting", "was", "better", "studies")) -> list[dict]:
    pp._nltk_pret()
    sortie = []
    for mot in mots:
        contexte = {"was": ["it", "was", "good"], "better": ["a", "better", "film"], "acting": ["he", "was", "acting"]}.get(mot, ["the", mot])
        sortie.append({
            "mot": mot,
            "lemme": pp.lemmatiser(contexte)[contexte.index(mot)],
            "racine": pp.raciniser([mot])[0],
        })
    return sortie


# --- Validation croisée du pipeline A -----------------------------------------------------------

def creer_modele_reference() -> Pipeline:
    """Modèle fixe servant uniquement à comparer les prétraitements (réglé en phase 5)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            token_pattern=pp.MOTIF_VECTORISEUR_A, lowercase=False, ngram_range=(1, 2),
            min_df=5, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=GRAINE)),
    ])


def evaluer_variante(textes: list[str], y: np.ndarray) -> dict:
    """Validation croisée stratifiée à 5 plis ; le vectoriseur est ajusté dans chaque pli."""
    plis = StratifiedKFold(n_splits=5, shuffle=True, random_state=GRAINE)
    debut = time.perf_counter()
    scores = cross_validate(
        creer_modele_reference(), np.array(textes, dtype=object), y, cv=plis,
        scoring=("accuracy", "f1", "roc_auc"), n_jobs=5,
    )
    return {
        "exactitude_par_pli": [round(float(v), 5) for v in scores["test_accuracy"]],
        "exactitude": round(float(scores["test_accuracy"].mean()), 5),
        "f1": round(float(scores["test_f1"].mean()), 5),
        "roc_auc": round(float(scores["test_roc_auc"].mean()), 5),
        "duree_s": round(time.perf_counter() - debut, 1),
    }


def decider(reference: dict, variante: dict) -> dict:
    """Une variante n'est retenue que si son gain apparié dépasse le bruit entre plis."""
    ecarts = np.array(variante["exactitude_par_pli"]) - np.array(reference["exactitude_par_pli"])
    gain = float(ecarts.mean())
    erreur_type = float(ecarts.std(ddof=1) / math.sqrt(len(ecarts)))
    return {
        "gain_exactitude_points": round(100 * gain, 3),
        "erreur_type_points": round(100 * erreur_type, 3),
        "plis_gagnants": int((ecarts > 0).sum()),
        "ameliore": bool(gain >= SEUIL_GAIN_MINIMAL and gain > 2 * erreur_type),
    }


# --- Orchestration ----------------------------------------------------------------------------

def _lire_json(nom: str) -> dict | None:
    chemin = DOSSIER_CACHE / nom
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else None


def _ecrire_json(nom: str, contenu: dict) -> None:
    DOSSIER_CACHE.mkdir(parents=True, exist_ok=True)
    (DOSSIER_CACHE / nom).write_text(json.dumps(contenu, ensure_ascii=False, indent=2), encoding="utf-8")


def chronometrer(textes: list[str], n_total: int) -> dict:
    """Mesure sur 2 000 critiques, sur un seul processus, puis extrapole à tout le corpus."""
    resultats = {}
    pp._nltk_pret()
    etapes = {
        "tronc_commun": lambda t: pp.tronc_commun(t),
        "pipeline_a_sans_normalisation": lambda t: pp.pipeline_a(t),
        "racinisation_porter": lambda t: pp.pipeline_a(t, normalisation="racine"),
        "lemmatisation_wordnet_pos": lambda t: pp.pipeline_a(t, normalisation="lemme"),
    }
    for nom, fonction in etapes.items():
        pp._etat_nltk["cache_lemmes"].clear()
        pp._etat_nltk["cache_racines"].clear()
        debut = time.perf_counter()
        for texte in textes:
            fonction(texte)
        duree = time.perf_counter() - debut
        resultats[nom] = {
            "secondes_sur_2000": round(duree, 2),
            "extrapolation_minutes_49575_un_coeur": round(duree / len(textes) * n_total / 60, 2),
        }
    return resultats


def executer(recalculer: bool = False) -> dict:
    """Enchaîne tout le travail de la phase 2 et met à jour reports/metrics.json."""
    pp._nltk_pret()
    DOSSIER_CACHE.mkdir(parents=True, exist_ok=True)
    corpus = dedoublonner(charger_brut(), mode="sans_ponctuation")
    entrainement, test = separer_entrainement_test(corpus, graine=GRAINE)
    textes_train = entrainement["review"].tolist()
    y_train = (entrainement["sentiment"] == "positive").astype(int).to_numpy()
    print(f"Corpus {len(corpus)} ; entraînement {len(entrainement)} ; test {len(test)} (non lu).")

    # 1. Chronométrage préalable sur 2 000 critiques d'entraînement, tirées avec la graine 42
    chrono = None if recalculer else _lire_json("chrono.json")
    if chrono is None:
        echantillon = entrainement.groupby("sentiment", group_keys=False).sample(n=1000, random_state=GRAINE)
        chrono = chronometrer(echantillon["review"].tolist(), len(corpus))
        _ecrire_json("chrono.json", chrono)
    print("Chronométrage :", chrono)

    # 2. Variantes du pipeline A, sur l'entraînement seulement
    chemin_variantes = DOSSIER_CACHE / "variantes_a_train.parquet"
    if chemin_variantes.exists() and not recalculer:
        variantes = pd.read_parquet(chemin_variantes)
    else:
        variantes = pd.DataFrame({"id_brut": entrainement["id_brut"].to_numpy()})
        for nom, reglages in VARIANTES_A.items():
            debut = time.perf_counter()
            variantes[nom] = _en_parallele(_lot_variante, textes_train, reglages)
            print(f"  variante {nom} : {time.perf_counter() - debut:.1f} s")
        variantes.to_parquet(chemin_variantes, index=False)

    # 3. Validation croisée
    cv = None if recalculer else _lire_json("validation_croisee.json")
    if cv is None:
        cv = {}
        for nom in VARIANTES_A:
            cv[nom] = evaluer_variante(variantes[nom].tolist(), y_train)
            print(f"  CV {nom} : {cv[nom]['exactitude']:.5f} en {cv[nom]['duree_s']} s")
        _ecrire_json("validation_croisee.json", cv)
    decisions = {nom: decider(cv["reference"], cv[nom]) for nom in VARIANTES_A if nom != "reference"}
    print("Décisions :", json.dumps(decisions, ensure_ascii=False))
    # Effet propre des négations : même retrait des mots vides, avec ou sans « not, no, nor, never »
    effet_negations = decider(cv["mots_vides_retires"], cv["mots_vides_sauf_negations"])
    print("Effet de garder les négations :", effet_negations)

    # 4. Lemmatisation contre racinisation
    comparaison = comparer_lemme_racine(
        variantes["reference"].tolist(), variantes["lemmatisation"].tolist(), variantes["racinisation"].tolist()
    )
    comparaison["exemples"] = exemples_lemme_racine()

    # 5. Ressources du pipeline B, calculées sans les étiquettes (usage descriptif)
    vides = _mots_vides_anglais() | frozenset(FORMES_SANS_APOSTROPHE)
    chemin_seq = DOSSIER_CACHE / "sequences_b.parquet"
    if chemin_seq.exists() and not recalculer:
        sequences = pd.read_parquet(chemin_seq)["sequence"].tolist()
    else:
        debut = time.perf_counter()
        sequences = _en_parallele(_lot_sequences, corpus["review"].tolist(), vides)
        print(f"  séquences B : {time.perf_counter() - debut:.1f} s")
        pd.DataFrame({"id_brut": corpus["id_brut"], "sequence": sequences}).to_parquet(chemin_seq, index=False)

    mots_domaine, info_domaine = choisir_mots_domaine(sequences)
    bigrammes, info_bigrammes = choisir_bigrammes(sequences, set(mots_domaine))

    # Contrôle de stabilité : les mêmes règles, appliquées à l'entraînement seul
    masque_train = corpus["id_brut"].isin(entrainement["id_brut"]).to_numpy()
    seq_train = [s for s, garde in zip(sequences, masque_train) if garde]
    domaine_train, _ = choisir_mots_domaine(seq_train)
    # Le seuil d'occurrences suit la taille de l'échantillon, pour comparer à règle équivalente.
    seuil_train = round(BIGRAMME_MIN_OCCURRENCES * len(seq_train) / len(sequences))
    bigrammes_train, _ = choisir_bigrammes(seq_train, set(domaine_train), seuil_train)
    jaccard = lambda a, b: round(len(set(a) & set(b)) / len(set(a) | set(b)), 4)
    stabilite = {
        "jaccard_mots_domaine_corpus_vs_entrainement": jaccard(mots_domaine, domaine_train),
        "jaccard_bigrammes_corpus_vs_entrainement": jaccard(bigrammes, bigrammes_train),
    }

    pp.CHEMIN_RESSOURCES_B.parent.mkdir(parents=True, exist_ok=True)
    pp.CHEMIN_RESSOURCES_B.write_text(json.dumps({
        "description": "Ressources du pipeline B, calculées sur le corpus dédoublonné sans utiliser les étiquettes.",
        "mots_vides_anglais": sorted(vides),
        "mots_vides_domaine": mots_domaine,
        "bigrammes": bigrammes,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    pp._ressources_b = None

    # 6. Textes finaux en cache
    ens_domaine = frozenset(mots_domaine)
    ens_bigrammes = frozenset(tuple(b.split("_")) for b in bigrammes)
    textes_b = [appliquer_ressources(s, ens_domaine, ens_bigrammes) for s in sequences]
    reglages_a = choisir_reglages_a(decisions)
    if reglages_a == {}:
        textes_a = _en_parallele(_lot_variante, corpus["review"].tolist(), {})
    else:
        textes_a = _en_parallele(_lot_variante, corpus["review"].tolist(), reglages_a)
    partition = np.where(masque_train, "entrainement", "test")
    pd.DataFrame({
        "id_brut": corpus["id_brut"].to_numpy(), "sentiment": corpus["sentiment"].to_numpy(),
        "partition": partition, "texte_a": textes_a, "texte_b": textes_b,
    }).to_parquet(CHEMIN_TEXTES, index=False)

    # Effet sur le pipeline B (avant = lemmes sans mots vides du domaine)
    occurrences_avant = sum(sum(1 for m in s.split() if m != SEPARATEUR) for s in sequences)
    occurrences_apres = sum(len(t.split()) for t in textes_b)
    vides_b = sum(1 for t in textes_b if not t)

    bilan = {
        "separation": {
            "methode": "80/20 stratifié, graine 42, après dédoublonnage",
            "entrainement": int(len(entrainement)), "test": int(len(test)),
            "note": (
                "Les choix du pipeline A ne lisent que l'entraînement. Les listes du pipeline B "
                "utilisent tout le corpus dédoublonné, sans étiquettes, pour un usage descriptif."
            ),
        },
        "outils": {"nltk": __import__("nltk").__version__, "lemmatiseur": "WordNet + étiquettes grammaticales", "raciniseur": "Porter"},
        "nettoyage": mesurer_nettoyage(corpus),
        "negations_entrainement": compter_negations(textes_train, variantes["reference"].tolist()),
        "chronometrage": chrono,
        "lemmatisation_contre_racinisation": comparaison,
        "validation_croisee_a": {
            "modele_de_comparaison": "TF-IDF (1,2), min_df=5, sublinear_tf, régression logistique C=1",
            "plis": 5,
            "regle": "retenue si gain moyen apparié >= 0,05 point et > 2 erreurs-types",
            "resultats": cv,
            "decisions": decisions,
            "effet_garder_negations": effet_negations,
            "reglages_retenus": reglages_a,
        },
        "pipeline_b": {
            "mots_vides_domaine": info_domaine,
            "liste_retenue": mots_domaine,
            "bigrammes": info_bigrammes,
            "stabilite": stabilite,
            "occurrences_avant_mots_domaine": occurrences_avant,
            "occurrences_apres": occurrences_apres,
            "critiques_vides_apres_pipeline_b": vides_b,
        },
    }
    enregistrer_metrics("pretraitement", bilan)
    return bilan


def choisir_reglages_a(decisions: dict) -> dict:
    """Traduit les décisions en réglages. Une normalisation n'est retenue que si elle gagne."""
    reglages: dict = {}
    if decisions["ponctuation"]["ameliore"]:
        reglages["ponctuation"] = True
    candidates = [
        (decisions[nom]["gain_exactitude_points"], valeur)
        for nom, valeur in (("lemmatisation", "lemme"), ("racinisation", "racine"))
        if decisions[nom]["ameliore"]
    ]
    if candidates:
        reglages["normalisation"] = max(candidates)[1]
    return reglages


if __name__ == "__main__":
    analyseur = argparse.ArgumentParser(description="Mesures et caches de la phase 2.")
    analyseur.add_argument("--recalculer", action="store_true", help="Ignore les caches existants.")
    executer(recalculer=analyseur.parse_args().recalculer)

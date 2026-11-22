"""Prépare, sur l'ordinateur, le seul fichier de données à déposer sur Google Drive.

Le fichier reprend les critiques dédoublonnées et les prétraitements déjà validés du projet.
`data/raw/` n'est jamais modifié, et ce fichier .parquet n'est pas suivi par git.

Exemple d'utilisation (depuis la racine du projet) :
    .\\.venv\\Scripts\\python.exe colab\\preparer_entree.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_COLAB = Path(__file__).resolve().parent / "projet4_colab"
sys.path[:0] = [str(RACINE), str(DOSSIER_COLAB)]

from src import bertopic_model as bm  # noqa: E402
from src import features as f  # noqa: E402
from src import topics as tp  # noqa: E402
from src.data import SHA256_BRUT, charger_brut  # noqa: E402
from themes_corpus_complet import NOM_ENTREE, empreinte_contenu  # noqa: E402


def preparer() -> Path:
    """Assemble texte léger, texte du pipeline B, étiquette et vocabulaire des thèmes."""
    table = pd.read_parquet(f.CHEMIN_TEXTES, columns=["id_brut", "sentiment", "texte_b"]).sort_values("id_brut")
    matrices = tp.charger_matrices_topics()
    if set(table["id_brut"]) != set(matrices["id_brut"].tolist()):
        raise ValueError("Les critiques ne correspondent pas à celles des modèles de thèmes.")
    brut = charger_brut()[["id_brut", "review"]].set_index("id_brut")
    legers = [bm.nettoyer_leger(t).replace("\n", " ") for t in brut.loc[table["id_brut"], "review"]]
    if not all(legers):
        raise ValueError("Une critique est vide après le nettoyage léger.")
    ids = table["id_brut"].to_numpy(dtype=np.int64)
    textes_b = table["texte_b"].tolist()
    manifeste = {
        "critiques": len(ids),
        "empreinte": empreinte_contenu(ids, legers, textes_b),
        "sha256_csv_brut": SHA256_BRUT,
        "pretraitement": "HTML, adresses et notes écrites retirés ; casse, ponctuation et mots vides conservés",
        "termes": list(matrices["termes"]),
    }
    sortie = pa.table({
        "id_brut": pa.array(ids, type=pa.int64()),
        "sentiment": pa.array(table["sentiment"].tolist(), type=pa.string()),
        "texte_leger": pa.array(legers, type=pa.string()),
        "texte_b": pa.array(textes_b, type=pa.string()),
    }).replace_schema_metadata({b"projet4": json.dumps(manifeste, ensure_ascii=False).encode("utf-8")})
    chemin = DOSSIER_COLAB / NOM_ENTREE
    pq.write_table(sortie, chemin, compression="zstd", compression_level=9)
    print(f"{chemin} : {len(ids)} critiques, {chemin.stat().st_size / 1e6:.1f} Mo")
    return chemin


if __name__ == "__main__":
    preparer()

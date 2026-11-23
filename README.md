# Critiques IMDB : thèmes et sentiment

Projet de Web Mining (ISSEA, ISE3-DSM). Quels thèmes reviennent dans les critiques favorables ou défavorables, et peut-on prédire le sentiment d'une nouvelle critique ? Le dépôt contient l'application Streamlit, le code, les chiffres, les figures et la présentation.

- **Application en ligne :** [https://web-mining-imdb-issea.streamlit.app](https://web-mining-imdb-issea.streamlit.app)
- **Auteurs :** [noms à compléter]

## Résultats principaux (source : `reports/metrics.json`)
- Modèle retenu : SVM linéaire calibré. Exactitude de **92,05 %** sur les 9 915 critiques de test, utilisées une seule fois.
- Thèmes : NMF à 8 thèmes (modèle commun). LDA et BERTopic sont comparés dans l'application, avec leurs résultats précalculés.
- Entre 0,28 et 0,72, l'application répond « incertain » au lieu de trancher.

## Ce que contient le dépôt
| Dossier | Contenu |
|---|---|
| `app/` | Application Streamlit, 6 pages : Accueil, Analyse descriptive, Exploration des thèmes, Performances des modèles, Analyse d'une nouvelle critique, Méthodologie et limites (point d'entrée `app/app.py`) et ses dépendances légères `app/requirements.txt` |
| `src/` | Code : prétraitement, vectorisation, thèmes, modèles, évaluation |
| `models/deploy/` | SVM calibré, NMF et ressources WordNet embarquées, avec leurs notices de licence |
| `notebooks/` | Notebook du projet, publié **sans sorties** (voir plus bas) |
| `reports/` | `metrics.json` (source unique des chiffres) et `figures/` |
| `livrables/` | Présentation du projet (PowerPoint). La vidéo de démonstration de l'application y sera ajoutée |
| `colab/` | Scripts du calcul BERTopic sur le corpus entier (Google Colab, BGE-M3), sans les résultats bruts |
| `tests/` | Tests `pytest`, dont ceux de l'application (`AppTest`) |
| `.streamlit/config.toml` | Thème de l'application |

Le corpus n'est pas dans le dépôt : un lecteur doit le télécharger lui-même (voir « Données »).

## Lancer l'application en local (plan B, une commande)
Sous Windows, avec Python 3.12 (lanceur `py`), depuis la racine du dépôt :

```powershell
.\lancer_application.ps1
```

Au premier lancement, le script crée `.venv` et installe `app/requirements.txt`. L'application s'ouvre ensuite sur http://localhost:8501. Le premier chargement des modèles prend quelques secondes.

## Installation complète et tests
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```
Les tests qui exigent le corpus, ses caches ou l'historique du dépôt de travail sont ignorés (`skipped`) ici. `requirements-avance.txt` ne sert qu'à BERTopic (torch) et n'est pas utilisé par l'application.

## Données
- Corpus : Kaggle, *IMDB Dataset of 50K Movie Reviews* (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). À placer dans `data/raw/IMDB Dataset.csv` pour rejouer le notebook et les scripts de `src/`.
- **Licence du corpus : [à vérifier].** Kaggle renvoie à une description qui n'en précise aucune. Détails dans `SOURCES.md`.
- L'application affiche une critique entière et 8 courts extraits (160 caractères au plus), listés dans `docs/extraits_application.md`.

## Notebook
`notebooks/projet4_imdb.ipynb` est publié sans sorties, car celles-ci reproduisaient des extraits de critiques. Pour le rejouer, placer le corpus puis exécuter le notebook : les sorties se régénèrent.

## Déployer sur Streamlit Community Cloud
1. Sur https://share.streamlit.io, se connecter et autoriser GitHub.
2. « Create app », puis « Yup, I have an app ».
3. Choisir ce dépôt, la branche, et le fichier principal `app/app.py`.
4. Dans « Advanced settings », choisir Python 3.12 (impossible à changer après coup sans supprimer l'application).
5. « Deploy ». Le fichier `app/requirements.txt` est lu en priorité sur celui de la racine.

- **Mettre à jour :** pousser sur la branche déployée. Un changement de dépendances relance l'installation.
- **Réveiller :** sans visite pendant 12 heures, l'application s'endort. Ouvrir l'adresse et cliquer sur « Yes, get this app back up! ». Prévoir de l'ouvrir avant la soutenance.
- **Redémarrer :** « Manage app », menu à trois points, « Reboot app ».
- **Supprimer :** « Manage app », menu à trois points, « Delete app ».

## Limites
- Le modèle a appris sur des critiques entières en anglais. Sur une phrase seule, un texte dans une autre langue, un avis neutre ou du sarcasme, il peut se tromper : l'application le signale.
- La séparation entraînement/test est aléatoire, sans identifiant de film : le score de test peut être un peu optimiste.
- Mémoire mesurée sous Windows en local seulement (pic de 619,9 Mo), avant la refonte de l'interface : [à vérifier] sur Streamlit Community Cloud.

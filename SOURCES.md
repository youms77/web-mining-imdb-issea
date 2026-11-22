# Sources des données

Vérification en ligne effectuée le 21/11/2026.

| Source | Adresse | Ce qui a été vérifié | Version et date | Licence |
|---|---|---|---|---|
| Kaggle, *IMDB Dataset of 50K Movie Reviews* (propriétaire : Lakshmipathi N) | https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews | Fichier `IMDB Dataset.csv` (66,21 Mo), colonnes `review` et `sentiment`, 49 582 valeurs uniques de `review`, renvoi vers la page de Stanford | « Updated 8 years ago » ; date exacte et numéro de version [à vérifier] | Affichée : « Other (specified in description) ». La description ne précise aucune licence : **[à vérifier]** |
| Stanford, *Large Movie Review Dataset* (aclImdb) | https://ai.stanford.edu/~amaas/data/sentiment/ | 25 000 critiques d'entraînement, 25 000 de test, 50 000 non étiquetées ; seuils ≤ 4 et ≥ 7 | Archive `aclImdb_v1.tar.gz`, non téléchargée | Aucune licence propre au corpus sur la page : **[à vérifier]** |
| Maas, Daly, Pham, Huang, Ng et Potts (2011), *Learning Word Vectors for Sentiment Analysis*, ACL-HLT, Portland, p. 142-150 | https://aclanthology.org/P11-1015/ | Section 4.3.2, p. 149 : 50 000 critiques, 30 au plus par film, classes égales, seuils ≤ 4 et ≥ 7, avis neutres exclus. Fin de la section 4.3.1, p. 148-149 : films disjoints entre entraînement et test | Actes publiés en juin 2011 | Article consulté, non redistribué |

## Fichier utilisé
- `data/raw/IMDB Dataset.csv`, fourni par l'utilisateur et jamais modifié.
- SHA256 : `DFC447764F82BE365FA9C2BEEF4E8DF89D3919E3DA95F5088004797D79695AA2`.

## Ressources linguistiques (phase 2)

Téléchargées le 21/11/2026 avec `nltk.download`, dans `data/interim/nltk_data/` (non versionné). Recréation : `installer_ressources_nltk()` de `src/preprocessing.py`.

| Ressource | Adresse | Version | Licence |
|---|---|---|---|
| NLTK | https://www.nltk.org/ | 3.10.4 | Apache 2.0 (métadonnées du paquet) |
| WordNet | paquet NLTK `wordnet` | WordNet 3.0, SHA256 de `wordnet.zip` : `CBDA5EA6EEF7F36A97A43D4A75F85E07FCCBB4F23657D27B4CCBC93E2646AB59` | Licence Princeton lue dans le paquet : usage et redistribution permis si la notice de copyright accompagne chaque copie |
| Étiqueteur grammatical | paquet NLTK `averaged_perceptron_tagger_eng` | SHA256 du zip : `6025F530624335C67D6547D44757B357B4E79BAE030A0383E9887A92C1718F0B` | MIT selon l'index officiel `nltk_data`, lu le 22/11/2026. Copyright 2013 Matthew Honnibal et 2015 The NLTK Project (en-tête de `nltk/tag/perceptron.py`). Corpus d'entraînement des poids : **[à vérifier]** |
| Mots vides anglais | paquet NLTK `stopwords` | Liste de 198 mots, recopiée dans `src/ressources/pipeline_b.json` | « Unknown » dans l'index `nltk_data`, lu le 22/11/2026 : **[à vérifier]** |

## Référence de méthode (phase 4)

| Référence | Adresse | Ce qui a été vérifié | Date | Licence |
|---|---|---|---|---|
| Monroe, Colaresi et Quinn (2008), *Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict*, Political Analysis 16(4), p. 372-403, DOI 10.1093/pan/mpn018 | https://languagelog.ldc.upenn.edu/myl/Monroe.pdf | Équations 16 (log-odds), 19 (variance), 22 (score z), p. 385-386 ; équation 23 (a priori informatif), p. 388 ; note 18 sur le double usage des données, p. 387 | Consulté le 21/11/2026 | Article consulté, non redistribué |

Bibliothèque ajoutée en phase 4 : gensim 4.4.0 (LDA et cohérence), épinglée dans `requirements.txt`.

## Référence lexicale (phase 5)

| Ressource | Adresse | Ce qui a été vérifié | Version et date | Licence |
|---|---|---|---|---|
| vaderSentiment (C.J. Hutto), paquet Python | https://github.com/cjhutto/vaderSentiment | Fichier `LICENSE.txt` lu dans le paquet avant l'installation. SHA256 du wheel : `3BF1D243B98B1AFAD575B9F22BC2CB1E212B94FF89CA74F8A23A588D024EA311` | 3.3.2, téléchargée le 22/11/2026 depuis PyPI, épinglée dans `requirements.txt` | MIT (copyright 2016 C.J. Hutto) |
| Hutto et Gilbert (2014), *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*, ICWSM-14, Ann Arbor | Citation demandée dans l'en-tête de `vaderSentiment.py` | Référence relevée dans le code du paquet. Article non lu | Juin 2014 | Article non redistribué |

VADER a été conçu pour les textes de réseaux sociaux : il est utilisé ici comme simple point de comparaison.

## Modèle d'embeddings et BERTopic (phase 6)

| Ressource | Adresse | Ce qui a été vérifié | Version et date | Licence |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (sentence-transformers) | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Longueur maximale de 256 jetons, dimension 384. Poids en safetensors, SHA256 : `53AA51172D142C89D9012CCE15AE4D6CC0CA6895895114379CACB4FAB128D9DB` | Révision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, téléchargée le 22/11/2026 | Apache-2.0 (carte du modèle) |
| BERTopic | https://github.com/MaartenGr/BERTopic | Signatures lues dans la version installée | 0.17.4, PyPI, 22/11/2026 | MIT (métadonnées du paquet) |
| sentence-transformers, torch (CPU), umap-learn, hdbscan | PyPI | Versions dans `requirements-avance.txt` | 6.3.0, 2.15.0, 0.6.0, 0.8.44 | Apache-2.0, licences multiples dont BSD, BSD, BSD (métadonnées) |
| Seuil neutre de VADER (±0,05) | https://github.com/cjhutto/vaderSentiment (README.rst) | Lignes 243 à 245 du README | Consulté le 22/11/2026 | MIT |

## Application (phase 7)

| Ressource | Adresse | Ce qui a été vérifié | Version et date | Licence |
|---|---|---|---|---|
| Streamlit | https://streamlit.io/ | Signatures lues dans la version installée | 1.67.0, PyPI, 22/11/2026 | Apache-2.0 (métadonnées du paquet) |
| Plotly | https://plotly.com/python/ | Version déjà présente dans `.venv` | 7.1.0 | MIT (métadonnées du paquet) |
| WordNet et étiqueteur embarqués | `models/deploy/nltk_data/` | Copies identiques aux empreintes ci-dessus. Notice Princeton et mentions MIT reproduites dans `models/deploy/nltk_data/LICENCES.md`. Le paquet `stopwords` n'est pas embarqué | 10,28 Mo et 1,47 Mo | Voir « Ressources linguistiques » |

Textes du corpus affichés par l'application : liste complète dans `docs/extraits_application.md`, pour la décision de publication de la phase 8.

## Points non établis
- Licence de la liste de mots vides NLTK et corpus d'entraînement des poids de l'étiqueteur : à contrôler avant le déploiement public.
- Conditions exactes de réutilisation du corpus. Par prudence, le CSV n'est pas versionné ni redistribué. L'application affiche une critique entière et 8 courts extraits, listés pour décision.
- Correspondance ligne à ligne entre le CSV Kaggle et l'archive de Stanford : non contrôlée, car l'archive n'a pas été téléchargée.
- Le README annoncé par la page de Stanford répond 404 à l'adresse `https://ai.stanford.edu/~amaas/data/sentiment/README` (21/11/2026).

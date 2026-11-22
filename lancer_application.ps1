# Plan B de la soutenance : lance l'application en local, en une commande.
# Utilisation (PowerShell, depuis la racine du dépôt) :  .\lancer_application.ps1
# Crée l'environnement .venv (Python 3.12) au premier lancement, puis démarre l'application.
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r app/requirements.txt
}
.\.venv\Scripts\python.exe -m streamlit run app/app.py

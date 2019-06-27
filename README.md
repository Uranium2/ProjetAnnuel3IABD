# ProjetAnnuel3IABD
Projet annuel de la classe IABD 3. 2018-2019

## Membres

Uranium2 - TAVERNIER Antoine

LittleSoap - ALLEXANDRE Matthieu

stephaneArtist - HOLLANDER Stéphane

# Classification

- Application permettant de différencier une image (screenshot) issue d'un RTS, d'un MOBA ou d'un FPS

# Environnement pour Python (Test case + Web)

Créer un environnement python (windows):

    py -m venv .venv
    .venv\Scripts\activate.bat
    
Charger tous les packages requis:

    pip install -r requirements.txt
   
Pour quitter l'environnement:

    deactivate

# Generate DLL

Avant de lancer les cas de tests, lancez la solution du projet pour générer la dll

# Test Case

Pour lancer un test case, il faut se positionner dans le dossier `PyTest`, puis lancer fichier test avec python:

    cd PyTest\
    py linear_classif_and.py

# Web

Lancer le back/front en Flask

    py web\app.py
    
Ouvrir le serveur Web:

    http://127.0.0.1:5000/
    


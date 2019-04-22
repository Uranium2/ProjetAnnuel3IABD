# ProjetAnnuel3IABD
Projet annuel de la classe IABD 3. 2018-2019

## Membres

Uranium2 - TAVERNIER Antoine

LittleSoap - ALLEXANDRE Matthieu

stephaneArtist - HOLLANDER Stéphane

# Classification


- Application permettant de différencier une image (screenshot) issue d'un RTS, d'un MOBA ou d'un FPS

Support Application:
  Unity (Android) ou alors Web


Plateforme pour coder: Visual Studio community 2019 - Windows 10
Lib: OpenCV + Eigen

Méthode pour récupérer les images d'un stream:

  - https://streamlink.github.io/install.html
  - https://www.youtube.com/watch?v=geF_i71I-ZM
  
Méthode pour récupérer les images sur Google image:
  - https://github.com/hardikvasa/google-images-download

______________________________________________________________________________

Environnement pour le scrapping d'image.

Python3 3.6.7

Créer un environnement python (windows):

    py -m pip install --user virtualenv
    py -m virtualenv env_folder
    source env_folder/bin/activate

Pour charger tous les packages requis:
  
    pip install -r requirements.txt
    
Pour quitter l'environnement:

    deactivate
    
# Notes

Avoir un "petit" dataset au début. 3 jeux par classe. Les normaliser (ratio, resolution). Garder le set brut et tester aussi avec du grayscale ou edge detect. Resultat possible, apprentissage rapide avec du preprocess, mais surment moins précis si on apprend plus longtemps.

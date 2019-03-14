# ProjetAnnuel3IABD
Projet annuel de la classe IABD 3. 2018-2019

## Membres

Uranium2 - TAVERNIER Antoine
LittleSoap - ALLEXANDRE Matthieu
stephaneArtist - HOLLANDER Stéphane

# Classification

- Application permettant de catégoriser une image (photo) soit en classe 'Chat' soit en classe 'Chien' soit
'autre'

- Application permettant de différencier une image (screenshot) issue d'un RTS, d'un MOBA ou d'un FPS

Régression :

- Application permettant de prédire l'âge d'une personne à partir d'une photo de son visage (attention à la
constitution du dataset)

Support Application:
  Unity (Android) ou alors Web


Plateforme pour coder: Windows (pas de VM ou de Linux, pb de 'driver' ?) + Visual Studio Code.

Question au prof: Demander si les lettres manuscrittes sont une lettre par une lettre, ou un mot/ article.
                Les datatsets des autres models?

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
    

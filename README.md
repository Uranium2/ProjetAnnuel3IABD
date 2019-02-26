# ProjetAnnuel3IABD
Projet annuel de la classe IABD 3. 2018-2019


Classification :

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

```python
from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

arguments = {"keywords":"League of Legends Gameplay",
                  "limit":100,
                  "prefix":"DOTA"}

paths = response.download(arguments)   #passing the arguments to the function
```

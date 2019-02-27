from google_images_download import google_images_download
import os
import glob
import cv2
import shutil

img_dir = "./downloads/" # Enter Directory of all images 
response = google_images_download.googleimagesdownload()

games = []
f = open("fps.txt", "r")
for x in f:
    games.append(x.rstrip())

i = 0
for game in games:
    arguments = {"keywords": game + " Gameplay",
                  "limit":5,
                  "prefix":"FPS" + str(i),
                  "format":"jpg"}
    i = i + 1
    paths = response.download(arguments)    # Download images
    source = img_dir + game + " Gameplay/"
    files = os.listdir(source)
    for f in files:                         # Move images to downloads dir
        shutil.move(source + f, img_dir)
    shutil.rmtree(source)


data_path = os.path.join(img_dir,'*')
files = glob.glob(data_path)
data = []
i = 0
for f1 in files:
    os.rename(f1, img_dir + "FPS_" + str('%04d' % i)) # Rename all files
    i = i + 1
    img = cv2.imread(f1)
    data.append(img)

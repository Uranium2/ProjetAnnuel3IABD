import tensorflow as tf
from tensorflow import keras
import numpy as np
from load_img import *
import matplotlib.pyplot as plt


img_per_folder = 1
h = 1
w = 1

#retrieve images
XTrain, Y = getDataSet("../img", img_per_folder, h, w, False)
XTrain = np.array(XTrain) 

print(XTrain)
print(Y)

#create models
W_FPS = tf.keras.Sequential([
			keras.layers.Flatten(input_shape=(28, 28, 1)),
			keras.layers.Dense(128, activation='tanh'),
			keras.layers.Dense(10, activation='softmax')
		])


W_MOBA = tf.keras.Sequential([
			keras.layers.Flatten(input_shape=(28, 28, 1)),
			keras.layers.Dense(128, activation='tanh'),
			keras.layers.Dense(10, activation='softmax')
		])


W_RTS = tf.keras.Sequential([
			keras.layers.Flatten(input_shape=(28, 28, 1)),
			keras.layers.Dense(128, activation='tanh'),
			keras.layers.Dense(10, activation='softmax')
		])

#compile models
W_FPS.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

W_MOBA.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

W_RTS.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

YTrain_FPS = []
YTrain_MOBA = []
YTrain_RTS = []

YTrain_FPS = np.array(YTrain_FPS) 
YTrain_MOBA = np.array(YTrain_MOBA) 
YTrain_RTS = np.array(YTrain_RTS) 

for i in range(img_per_folder * 3):
        if i < img_per_folder:
            YTrain_FPS.append(1)
        else:
            YTrain_FPS.append(-1)
for i in range(img_per_folder * 3):
    if i > img_per_folder and i < 2 * img_per_folder:
        YTrain_MOBA.append(1)
    else:
        YTrain_MOBA.append(-1)
for i in range(img_per_folder * 3):
    if i > 2 * img_per_folder and i < 3 * img_per_folder:
        YTrain_RTS.append(1)
    else:
        YTrain_RTS.append(-1)

YTrain_FPS = np.array(YTrain_FPS)
YTrain_MOBA = np.array(YTrain_MOBA)
YTrain_RTS = np.array(YTrain_RTS)

#Fit all models
W_FPS.fit(XTrain, YTrain_FPS, epochs=1)
W_MOBA.fit(XTrain, YTrain_MOBA, epochs=1)
W_RTS.fit(XTrain, YTrain_RTS, epochs=1)

#Evaluate models
Xpredict = []
imageToPredict = "../img/RTS_Test/RTS_0085.png"

size = inputCountPerSample / 3
size = int(math.sqrt( size ))


im = Image.open(imageToPredict)
im = im.convert("RGB")
imResize = im.resize((size, size), Image.ANTIALIAS)
imgLoad = imResize.load()

for x in range(size):
    for y in range(size):
        R,G,B = imgLoad[x, y]
        Xpredict.append(R / 255)
        Xpredict.append(G / 255)
        Xpredict.append(B / 255)
im.close()

Xpredict = np.array(Xpredict)

W_FPS.Evaluate(Xpredict)
W_MOBA.Evaluate(Xpredict)
W_RTS.Evaluate(Xpredict)
import numpy as np
from load_img import *
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import h5py

def linear_keras(img_per_folder, height, width, imageToPredict, batch_size=1):
   
    batch_size = 1
    inputCountPerSample = height * width * 3

    #retrieve images
    XTrain, Y = getDataSet("../img", img_per_folder, height, width, False)
    XTrain = np.array(XTrain)
    XTrain = XTrain.reshape(3 * img_per_folder, inputCountPerSample)
    # print(XTrain.shape)

    #create models
    W_FPS = Sequential()
    W_FPS.add(Dense(1, activation = 'tanh', input_dim = XTrain.shape[1]))

    W_MOBA = Sequential()
    W_MOBA.add(Dense(1, activation = 'tanh', input_dim = XTrain.shape[1]))

    W_RTS = Sequential()
    W_RTS.add(Dense(1, activation = 'tanh', input_dim = XTrain.shape[1]))

    #compile models

    W_FPS.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
    W_MOBA.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
    W_RTS.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

    YTrain_FPS = np.array([]) 
    YTrain_MOBA = np.array([]) 
    YTrain_RTS = np.array([]) 

    for i in range(img_per_folder * 3):
            if i < img_per_folder:
                YTrain_FPS = np.append(YTrain_FPS, 1)
            else:
                YTrain_FPS = np.append(YTrain_FPS, -1)
    for i in range(img_per_folder * 3):
        if i >= img_per_folder and i < 2 * img_per_folder:
            YTrain_MOBA = np.append(YTrain_MOBA, 1)
        else:
            YTrain_MOBA = np.append(YTrain_MOBA, -1)
    for i in range(img_per_folder * 3):
        if i >= 2 * img_per_folder and i < 3 * img_per_folder:
            YTrain_RTS = np.append(YTrain_RTS, 1)
        else:
            YTrain_RTS = np.append(YTrain_RTS, -1)


    #Fit all models
    W_FPS.fit(XTrain, YTrain_FPS, batch_size=batch_size, verbose=1, epochs=500)
    W_MOBA.fit(XTrain, YTrain_MOBA, batch_size=batch_size, verbose=1, epochs=500)
    W_RTS.fit(XTrain, YTrain_RTS, batch_size=batch_size, verbose=1, epochs=500)

    #save models   
    W_FPS.save("models/FPS_model")
    W_MOBA.save("models/MOBA_model")
    W_RTS.save("models/RTS_model")

    # Evaluate models
    Xpredict = []

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
    Xpredict = Xpredict.reshape(1, inputCountPerSample)
    # print(Xpredict)

    return W_FPS.predict(Xpredict), W_MOBA.predict(Xpredict) , W_RTS.predict(Xpredict)

print(linear_keras(50, 20, 20, "../img/RTS_Test/RTS_0140.png"))
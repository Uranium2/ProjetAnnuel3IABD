import h5py
import math
import numpy as np
from load_img import *
import matplotlib.pyplot as plt
from keras.models import load_model as load_model_tf
from keras.models import Sequential
from keras.layers import Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def predict_stat(model, isValidation, h=0, w=0):

    if(isValidation):
        img_per_folder = 200

    paths = getImgPath("../img", img_per_folder, h, w, isValidation)

def load_predict_mlp_tf(pathModel, imageToPredict):

    W_MLP = load_model_tf(model)

    # get image size from model
    size = inputCountPerSample / 3
    size = int(math.sqrt( size ))
    XTest = []
    Xpredict = []
    Ypredict = []

    # ouvre
    im = Image.open(imageToPredict)
    im = im.convert("RGB")
    imResize = im.resize((size, size), Image.ANTIALIAS)
    imgLoad = imResize.load()
    for x in range(size):
        for y in range(size):
            R, G, B = imgLoad[x, y]
            Xpredict.append(R)
            Xpredict.append(G)
            Xpredict.append(B)
    im.close()

    Ypredict.append(predict_mlp_regression(W, layers, layer_count, inputCountPerSample, Xpredict))
    Ypredict[0].pop(0)
    #print(Ypredict[0])

    index = Ypredict[0].index(max(Ypredict[0]))
    #print(index)
    return Ypredict, index

def mlp_model(img_per_folder, height, width, layers, imageToPredict, batch_size=3, epochs=100, W_MLP=None):

    inputCountPerSample = height * width * 3

    # retrieve images
    XTrain, Y = getDataSet("../img", img_per_folder, height, width, False)
    XTrain = np.array(XTrain)
    Y = np.array(Y)
    # print(XTrain)
    #print(Y)
    XTrain = XTrain.reshape(img_per_folder * 3, inputCountPerSample)
    Y = Y.reshape(img_per_folder * 3, 3)

    if W_MLP == None:

        # create model
        W_MLP = Sequential()
        for neuron in layers:
            W_MLP.add(Dense(neuron, activation='tanh', input_dim=inputCountPerSample))

        # compile model
        W_MLP.compile(optimizer='sgd', loss='mean_squared_error',
                      metrics=['accuracy'])

        # fit model
        W_MLP.fit(XTrain, Y, batch_size=batch_size, verbose=1, epochs=epochs)

        # save model
        W_MLP.save("models/MLP_Model")

    else:
        W_MLP = W_MLP

    # Evaluate model
    Xpredict = []

    size = inputCountPerSample / 3
    size = int(math.sqrt(size))

    im = Image.open(imageToPredict)
    im = im.convert("RGB")
    imResize = im.resize((size, size), Image.ANTIALIAS)
    imgLoad = imResize.load()

    for x in range(size):
        for y in range(size):
            R, G, B = imgLoad[x, y]
            Xpredict.append(R / 255)
            Xpredict.append(G / 255)
            Xpredict.append(B / 255)
    im.close()

    Xpredict = np.array(Xpredict)
    Xpredict = Xpredict.reshape(1, inputCountPerSample)

    return W_MLP.predict(Xpredict)


#print(mlp_model(500, 50, 50, [5, 5, 3], "../img/MOBA_Validation/MOBA_Validation_0004.png"))
model = load_model_tf("models/MLP_model")

for layer in model.layers:
    print(layer.input_shape)

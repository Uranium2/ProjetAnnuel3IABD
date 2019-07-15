from dll_load import (
    create_mlp_model,
    fit_mlp_classification,
    flatten,
    predict_mlp_regression,
    saveModel,
    loadModel,
)
from load_img import getDataSet, getImgPath, save_stats
from PIL import Image
import math
import h5py
import numpy as np
from load_img import *
import matplotlib.pyplot as plt
from keras.models import load_model as load_model_tf
from keras.models import Sequential
from keras.layers import Dense
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow.python.util.deprecation as deprecation
from keras import backend as K 

deprecation._PRINT_DEPRECATION_WARNINGS = False


def fit_save_mlp(img_per_folder, h, w, alpha, epochs, prefix, layers):
    oldLayers = layers.copy()
    inputCountPerSample = h * w * 3
    layers.append(3)
    layers.insert(0, inputCountPerSample)
    layer_count = len(layers)
    sampleCount = img_per_folder * 3

    XTrain, YTrain = getDataSet("../img", img_per_folder, h, w, False)

    W = create_mlp_model(layers, layer_count)

    fit_mlp_classification(
        W,
        XTrain,
        YTrain,
        layers,
        layer_count,
        sampleCount,
        inputCountPerSample,
        alpha,
        epochs,
    )
    file_return = prefix
    file_name = "Models/MLP/" + prefix
    for i in layers:
        file_name = file_name + "_" + str(i)
        file_return = file_return + "_" + str(i)
    file_name = file_name + ".model"
    file_return = file_return + ".model"

    saveModel(W, layers, layer_count, file_name)

    accuracy_Set = load_predict_mlp_stat(img_per_folder, file_name, False)
    accurracy_validation = load_predict_mlp_stat(img_per_folder, file_name, True)

    save_stats(
        "Multilayer perceptron : " + prefix,
        epochs,
        alpha,
        str(h) + "x" + str(w),
        img_per_folder * 3,
        oldLayers,
        accuracy_Set,
        accurracy_validation,
    )

    return file_return

def load_predict_mlp_stat(img_per_folder, pathModel, isValidation):
    if isValidation:
        img_per_folder = 200
    layer_count, layers, W = loadModel(pathModel)
    inputCountPerSample = layers[0]
    size = inputCountPerSample / 3
    size = int(math.sqrt(size))
    XTest = []
    Xpredict = []
    Ypredict = []

    files = getImgPath("../img", img_per_folder, size, size, isValidation)

    result = []
    for img in files:
        y, index = load_predict_mlp(pathModel, img)
        result.append(index)

    stat = []
    for i in range(len(result)):
        if result[i] == 0 and i < img_per_folder:
            stat.append(True)
        elif result[i] == 1 and i >= img_per_folder and i < 2 * img_per_folder:
            stat.append(True)
        elif result[i] == 2 and i >= 2 * img_per_folder and i < 3 * img_per_folder:
            stat.append(True)
        else:
            stat.append(False)

    return sum(stat) / len(stat) * 100

def load_predict_mlp(pathModel, imageToPredict):

    layer_count, layers, W = loadModel(pathModel)
    inputCountPerSample = layers[0]
    size = inputCountPerSample / 3
    size = int(math.sqrt(size))
    XTest = []
    Xpredict = []
    Ypredict = []

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

    Ypredict.append(
        predict_mlp_regression(W, layers, layer_count, inputCountPerSample, Xpredict)
    )
    Ypredict[0].pop(0)
    # print(Ypredict[0])

    index = Ypredict[0].index(max(Ypredict[0]))
    # print(index)
    return Ypredict, index

def load_mlp_model(model_path):
    K.clear_session()
    model = load_model_tf(model_path)
    return model


def mlp_keras(filename, img_per_folder, h, w, layers, epochs):
    layers.append(3)
    inputCountPerSample = h * w * 3
    file_name = "Models/MLPTF/" + filename
    file_return = filename
    file_name = file_name + "_" + str(inputCountPerSample)
    for i in layers:
        file_name = file_name + "_" + str(i)
        file_return = file_return + "_" + str(i)
    file_name = file_name + ".model"
    file_return = file_return + ".model"

    # retrieve images
    XTrain, Y = getDataSet("../img", img_per_folder, h, w, False)
    XTrain = np.array(XTrain)
    Y = np.array(Y)
    XTrain = XTrain.reshape(img_per_folder * 3, inputCountPerSample)
    Y = Y.reshape(img_per_folder * 3, 3)

    # create model
    W_MLP = Sequential()
    for neuron in layers:
        W_MLP.add(Dense(neuron, activation="tanh", input_dim=inputCountPerSample))

    # compile model
    W_MLP.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])

    # fit model
    W_MLP.fit(XTrain, Y, batch_size=1, verbose=1, epochs=epochs)

    # save model
    W_MLP.save(file_name)

    accuracy_Set = get_stats_mlp_tf(img_per_folder, file_name, False)
    accurracy_validation = get_stats_mlp_tf(img_per_folder, file_name, True)
    struct = layers[:-1]
    save_stats( "Multilayer perceptron Tensorflow : " + filename + "_" + str(inputCountPerSample), epochs, "", str(h) + "x" + str(w), img_per_folder * 3, struct, accuracy_Set, accurracy_validation)

    return file_return

def get_stats_mlp_tf(img_per_folder, model_path, isValidation):
    W_MLP = load_mlp_model(model_path)
    inputCountPerSample = int(model_path.split("_")[1])
    size = inputCountPerSample / 3
    size = int(math.sqrt(size))
    files = getImgPath("../img", img_per_folder, size, size, isValidation)
    result = []
    for img in files:
        res = predict_mlp_tf(W_MLP, img, inputCountPerSample)

        index = np.argmax(res)
        result.append(index)

    stat = []
    for i in range(img_per_folder * 3):
        if result[i] == 0 and i < img_per_folder:
            stat.append(True)
        elif result[i] == 1 and i >= img_per_folder and i < 2 * img_per_folder:
            stat.append(True)
        elif result[i] == 2 and i >= 2 * img_per_folder and i < 3 * img_per_folder:
            stat.append(True)
        else :
            stat.append(False)
    return (sum(stat)/ len(stat) * 100)

def web_predict_mlp_tf(model_path, imageToPredict):
    K.clear_session()
    W_MLP = load_mlp_model(model_path)
    inputCountPerSample = int(model_path.split("_")[1])
    return predict_mlp_tf(W_MLP, imageToPredict, inputCountPerSample)

def predict_mlp_tf(W_MLP, imageToPredict, inputCountPerSample):

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

    return W_MLP.predict(Xpredict)[0]
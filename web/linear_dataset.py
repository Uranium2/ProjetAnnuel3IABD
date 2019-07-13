from dll_load import create_linear_model, fit_classification_rosenblatt_rule, saveLinearModel, loadLinearModel, predict_regression
from load_img import getDataSet, getImgPath, save_stats
from PIL import Image
import math
import numpy as np
from load_img import *
import matplotlib.pyplot as plt
from keras.models import load_model as load_model_tf
from keras.models import Sequential
from keras.layers import Dense
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import h5py



def fit_save_classif(img_per_folder, h, w, alpha, epochs, prefix):

    inputCountPerSample = h * w * 3
    sampleCount = img_per_folder * 3


    XTrain, Y = getDataSet("../img", img_per_folder, h, w, False)

    W_FPS = create_linear_model(inputCountPerSample)
    W_MOBA = create_linear_model(inputCountPerSample)
    W_RTS = create_linear_model(inputCountPerSample)
    YTrain_FPS = []
    YTrain_MOBA = []
    YTrain_RTS = []

    for i in range(img_per_folder * 3):
        if i < img_per_folder:
            YTrain_FPS.append(1)
        else:
            YTrain_FPS.append(-1)
    for i in range(img_per_folder * 3):
        if i >= img_per_folder and i < 2 * img_per_folder:
            YTrain_MOBA.append(1)
        else:
            YTrain_MOBA.append(-1)
    for i in range(img_per_folder * 3):
        if i >= 2 * img_per_folder and i < 3 * img_per_folder:
            YTrain_RTS.append(1)
        else:
            YTrain_RTS.append(-1)

    fit_classification_rosenblatt_rule(W_FPS, XTrain, sampleCount, inputCountPerSample, YTrain_FPS, alpha, epochs)
    fit_classification_rosenblatt_rule(W_MOBA, XTrain, sampleCount, inputCountPerSample, YTrain_MOBA, alpha, epochs)
    fit_classification_rosenblatt_rule(W_RTS, XTrain, sampleCount, inputCountPerSample, YTrain_RTS, alpha, epochs)
    file_name_FPS = "Models\Linear\\" + prefix + "_FPS.model"
    file_name_MOBA = "Models\Linear\\" + prefix + "_MOBA.model"
    file_name_RTS = "Models\Linear\\" + prefix + "_RTS.model"
    saveLinearModel(W_FPS, inputCountPerSample, file_name_FPS)
    saveLinearModel(W_MOBA, inputCountPerSample, file_name_MOBA)
    saveLinearModel(W_RTS, inputCountPerSample, file_name_RTS)

    # Lancer un prédict sur le dataset de base + de validation + écrire dans le CSV
    accuracy_Set = load_predict_classif_stat(img_per_folder, file_name_FPS, file_name_MOBA, file_name_RTS, False)
    accurracy_validation = load_predict_classif_stat(img_per_folder, file_name_FPS, file_name_MOBA, file_name_RTS, True)

    save_stats( "Linear Model : " + prefix, epochs, alpha, str(h) + "x" + str(w), img_per_folder * 3, "", accuracy_Set, accurracy_validation)

    return  prefix + "_FPS.model", prefix + "_MOBA.model", prefix + "_RTS.model"

def load_predict_classif_stat(img_per_folder, pathFPS, pathMOBA, pathRTS, isValidation):
    if isValidation :
        img_per_folder = 200


    inputCountPerSample, WFPS = loadLinearModel(pathFPS)
    inputCountPerSample, WMOBA = loadLinearModel(pathMOBA)
    inputCountPerSample, WRTS = loadLinearModel(pathRTS)
    size = inputCountPerSample / 3
    size = int(math.sqrt( size ))

    files = getImgPath("../img", img_per_folder, size, size, isValidation)
    
    result = []
    for img in files:
        fps, moba, rts, index = load_predict_classif(pathFPS, pathMOBA, pathRTS, img)
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
    
def load_predict_classif(pathFPS, pathMOBA, pathRTS, imageToPredict):
    Xpredict = []

    Ypredict_FPS = []
    Ypredict_MOBA = []
    Ypredict_RTS = []


    inputCountPerSample, WFPS = loadLinearModel(pathFPS)
    inputCountPerSample, WMOBA = loadLinearModel(pathMOBA)
    inputCountPerSample, WRTS = loadLinearModel(pathRTS)

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

    Ypredict_FPS.append(predict_regression(WFPS, Xpredict, inputCountPerSample))
    Ypredict_MOBA.append(predict_regression(WMOBA, Xpredict, inputCountPerSample))
    Ypredict_RTS.append(predict_regression(WRTS, Xpredict, inputCountPerSample))


    for y in range(len(Ypredict_FPS)):
        if Ypredict_FPS[y] > Ypredict_MOBA[y] and Ypredict_FPS[y] > Ypredict_RTS[y]:
            return Ypredict_FPS ,Ypredict_MOBA , Ypredict_RTS, 0
        if Ypredict_MOBA[y] > Ypredict_FPS[y] and Ypredict_MOBA[y] > Ypredict_RTS[y]:
            return Ypredict_FPS ,Ypredict_MOBA , Ypredict_RTS, 1
        if Ypredict_RTS[y] > Ypredict_MOBA[y] and Ypredict_RTS[y] > Ypredict_FPS[y]:
            return Ypredict_FPS ,Ypredict_MOBA , Ypredict_RTS, 2
    return Ypredict_FPS ,Ypredict_MOBA , Ypredict_RTS, 3 # Error



def load_linear_model(model_path):
    model = load_model_tf(model_path)
    return model


def linear_keras(filename, img_per_folder, h, w, epochs):

    inputCountPerSample = h * w * 3

    # retrieve images
    XTrain, Y = getDataSet("../img", img_per_folder, h, w, False)
    XTrain = np.array(XTrain)
    XTrain = XTrain.reshape(3 * img_per_folder, inputCountPerSample)
    # print(XTrain.shape)

    # create models
    W_FPS = Sequential()
    W_FPS.add(Dense(1, activation="tanh", input_dim=XTrain.shape[1]))

    W_MOBA = Sequential()
    W_MOBA.add(Dense(1, activation="tanh", input_dim=XTrain.shape[1]))

    W_RTS = Sequential()
    W_RTS.add(Dense(1, activation="tanh", input_dim=XTrain.shape[1]))

    # compile models
    W_FPS.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])
    W_MOBA.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])
    W_RTS.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])

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

    # fit models
    W_FPS.fit(XTrain, YTrain_FPS, batch_size=1, verbose=1, epochs=epochs)
    W_MOBA.fit(XTrain, YTrain_MOBA, batch_size=1, verbose=1, epochs=epochs)
    W_RTS.fit(XTrain, YTrain_RTS, batch_size=1, verbose=1, epochs=epochs)

    # save models
    fps_path = "models/LinearTF/" + filename + "_" + str(inputCountPerSample) + "_FPS.model"
    W_FPS.save(fps_path)
    moba_path = "models/LinearTF/" + filename + "_" + str(inputCountPerSample) + "_MOBA.model"
    W_MOBA.save(moba_path)
    rts_path = "models/LinearTF/" + filename + "_" + str(inputCountPerSample) + "_RTS.model"
    W_RTS.save(rts_path)

    accuracy_Set = get_stats_linear_tf(img_per_folder, fps_path, moba_path, rts_path, False)
    accurracy_validation = get_stats_linear_tf(img_per_folder, fps_path, moba_path, rts_path, True)
    save_stats( "Linear Model : " + filename + "_" + str(inputCountPerSample), epochs, "", str(h) + "x" + str(w), img_per_folder * 3, "", accuracy_Set, accurracy_validation)

    return  filename + "_" + str(inputCountPerSample) + "_FPS.model", filename + "_" + str(inputCountPerSample) + "_MOBA.model", filename + "_" + str(inputCountPerSample) + "_RTS.model"


def get_stats_linear_tf(img_per_folder, pathFPS, pathMOBA, pathRTS, isValidation):
    FPS_model = load_linear_model(pathFPS)
    MOBA_model = load_linear_model(pathMOBA)
    RTS_model = load_linear_model(pathRTS)

    inputCountPerSample = int(pathFPS.split("_")[1])
    size = inputCountPerSample / 3
    size = int(math.sqrt(size))

    files = getImgPath("../img", img_per_folder, size, size, isValidation)
    result = []
    for img in files:
        fps, moba, rts, index = predict_linear_tf(FPS_model, MOBA_model, RTS_model, img, inputCountPerSample)
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
    
def web_predict_linear_tf(FPS_path, MOBA_path, RTS_path, imageToPredict, inputCountPerSample):
    FPS_model = load_linear_model("models/LinearTF/" + FPS_path)
    MOBA_model = load_linear_model("models/LinearTF/" + MOBA_path)
    RTS_model = load_linear_model("models/LinearTF/" + RTS_path)
    return predict_linear_tf(FPS_model, MOBA_model, RTS_model, imageToPredict, inputCountPerSample)

def predict_linear_tf(FPS_model, MOBA_model, RTS_model, imageToPredict, inputCountPerSample):
    # Evaluate models
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
    fps = FPS_model.predict(Xpredict)[0][0]
    moba = MOBA_model.predict(Xpredict)[0][0]
    rts = RTS_model.predict(Xpredict)[0][0]
    index = 0
    if fps > moba and fps > rts:
        index = 0
    if moba > fps and moba > rts:
        index = 1
    if rts > moba and rts > fps:
        index = 2
    return ( fps,
            moba,
            rts,
            index
    )

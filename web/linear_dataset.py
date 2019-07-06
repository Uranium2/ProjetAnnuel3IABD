from dll_load import create_linear_model, fit_classification_rosenblatt_rule, saveLinearModel, loadLinearModel, predict_regression
from load_img import getDataSet, getDataSetTest
from PIL import Image
import math


def fit_save_classif(img_per_folder, h, w, alpha, epochs, prefix):

    inputCountPerSample = h * w * 3
    sampleCount = img_per_folder * 3


    XTrain, Y = getDataSet("../img", img_per_folder, h, w)

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
        if i > img_per_folder and i < 2 * img_per_folder:
            YTrain_MOBA.append(1)
        else:
            YTrain_MOBA.append(-1)
    for i in range(img_per_folder * 3):
        if i > 2 * img_per_folder and i < 3 * img_per_folder:
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
    return  prefix + "_FPS.model", prefix + "_MOBA.model", prefix + "_RTS.model"

def load_predict_classif_stat(img_per_folder, h, w, pathFPS, pathMOBA, pathRTS):
    Xpredict = []

    Ypredict_FPS = []
    Ypredict_MOBA = []
    Ypredict_RTS = []

    inputCountPerSample, WFPS = loadLinearModel(pathFPS)
    inputCountPerSample, WMOBA = loadLinearModel(pathMOBA)
    inputCountPerSample, WRTS = loadLinearModel(pathRTS)

    XTest, Y = getDataSetTest("../img", img_per_folder, h, w)
    for img in range(img_per_folder * 3):
        for i in range(h * w * 3):
            Xpredict.append(XTest[img * i])
        Ypredict_FPS.append(predict_regression(WFPS, Xpredict, inputCountPerSample))
        Ypredict_MOBA.append(predict_regression(WMOBA, Xpredict, inputCountPerSample))
        Ypredict_RTS.append(predict_regression(WRTS, Xpredict, inputCountPerSample))
        Xpredict.clear()

    result = []
    for y in range(len(Ypredict_FPS)):
        if Ypredict_FPS[y] > Ypredict_MOBA[y] and Ypredict_FPS[y] > Ypredict_RTS[y]:
            result.append(0)
        if Ypredict_MOBA[y] > Ypredict_FPS[y] and Ypredict_MOBA[y] > Ypredict_RTS[y]:
            result.append(1)
        if Ypredict_RTS[y] > Ypredict_MOBA[y] and Ypredict_RTS[y] > Ypredict_FPS[y]:
            result.append(2)


    stat = []
    for i in range(img_per_folder):
        if result[i] == 0 and i < img_per_folder:
            stat.append(True)
        elif result[i + img_per_folder] == 1 and i > img_per_folder and i < 2 * img_per_folder:
            stat.append(True)
        elif result[i + (2 * img_per_folder)] == 2 and i > 2 * img_per_folder and i < 3 * img_per_folder:
            stat.append(True)
        else :
            stat.append(False)
    print( sum(stat)/ len(stat) * 100)
    
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
            Xpredict.append(R)
            Xpredict.append(G)
            Xpredict.append(B)
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


#fit_save_classif(1200, 100, 100, 0.05, 500, "100x100_1200_22h10")
#load_predict_classif_stat(100, "Models\\Linear\\linear_dataset_FPS_rendu3_18_30_1200.model",
#                            "Models\\Linear\\linear_dataset_MOBA_rendu3_18_30_1200.model",
#                            "Models\\Linear\\linear_dataset_RTS_rendu3_18_30_1200.model" )
from dll_load import (
    create_mlp_model,
    fit_mlp_classification,
    flatten,
    predict_mlp_regression,
    saveModel,
    loadModel,
)
from load_img import getDataSet
from PIL import Image
import math

def fit_save_mlp(img_per_folder, h, w, alpha, epochs, prefix, layers):
    inputCountPerSample = h * w * 3
    layers.append(3)
    layers.insert(0, inputCountPerSample)
    layer_count = len(layers)
    sampleCount = img_per_folder * 3

    XTrain, YTrain = getDataSet("../img", img_per_folder, h, w)

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
    file_name = "Models/MLP/" + prefix
    for i in layers:
        file_name = file_name + "_" + str(i)
    file_name = file_name + ".model"

    saveModel(W, layers, layer_count, file_name)


def load_predict_mlp_stat(img_per_folder, h, w, pathModel):
    layer_count, layers, W = loadModel(pathModel)
    XTest = []
    Xpredict = []
    Ypredict = []

    XTrain, Y = getDataSet("../img", img_per_folder, h, w)
    for img in range(img_per_folder * 3):
        for i in range(h * w * 3):
            Xpredict.append(XTrain[img * i])
        res = predict_mlp_regression(W, layers, layer_count, h * w * 3, Xpredict)
        res.pop(0)
        # print(res)
        Ypredict.append(res)
        Xpredict.clear()

    result = []
    for res3 in Ypredict:
        index = res3.index(max(res3))
        result.append(index)

    stat = []
    for i in range(len(result)):
        if result[i] == 0 and i < img_per_folder:
            stat.append(True)
        elif result[i] == 1 and i > img_per_folder and i < 2 * img_per_folder:
            stat.append(True)
        elif result[i] == 2 and i > 2 * img_per_folder and i < 3 * img_per_folder:
            stat.append(True)
        else:
            stat.append(False)

    print(sum(stat) / len(stat) * 100)


def load_predict_mlp(pathModel, imageToPredict):

    layer_count, layers, W = loadModel(pathModel)
    inputCountPerSample = layers[0]
    size = inputCountPerSample / 3
    size = int(math.sqrt( size ))
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

    Ypredict.append(predict_mlp_regression(W, layers, layer_count, inputCountPerSample, Xpredict))
    Ypredict[0].pop(0)
    print(Ypredict[0])

    index = Ypredict[0].index(max(Ypredict[0]))
    print(index)
    return Ypredict, index


# fit_save_mlp(50, 25, 25, 0.01, 1000, "Test_remove_me", [5, 5])

# load_predict_mlp_stat(10, 100, 100, "Models/MLP/Test_remove_me_1875_5_5_3.model")
# load_predict_mlp(
#     10,
#     100,
#     100,
#     "Models/MLP/Test_remove_me_1875_5_5_3.model",
#     "C:\\Users\\Tavernier\\Downloads\\RBF_naif.png",
# )

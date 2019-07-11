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

    save_stats( "Multilayer perceptron : " + prefix, epochs, alpha, str(h) + "x" + str(w), img_per_folder * 3, oldLayers, accuracy_Set, accurracy_validation)

    return file_return


def load_predict_mlp_stat(img_per_folder, pathModel, isValidation):
    if isValidation :
        img_per_folder = 50
    layer_count, layers, W = loadModel(pathModel)
    inputCountPerSample = layers[0]
    size = inputCountPerSample / 3
    size = int(math.sqrt( size ))
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

    return (sum(stat) / len(stat) * 100)



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
    #print(Ypredict[0])

    index = Ypredict[0].index(max(Ypredict[0]))
    #print(index)
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

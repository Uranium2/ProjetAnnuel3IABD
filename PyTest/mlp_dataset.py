from dll_load import create_mlp_model, fit_mlp_classification, flatten, predict_mlp_regression, saveModel, loadModel
from load_img import getDataSet
from PIL import Image


if __name__ == "__main__":
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

    def load_predict_mlp(img_per_folder, h, w, pathModel):
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
            print(res)
            Ypredict.append(res)
            Xpredict.clear()

        result = []

        for res3 in Ypredict:
            print(res3)
            if res3[0] > res[1] and res3[0] > res[2]:
                print(0)
                result.append(0)
            elif res3[1] > res[0] and res3[1] > res[2]:
                print(1)
                result.append(1)
            elif res3[2] > res[0] and res3[2] > res[1]:
                print(2)
                result.append(2)

        stat = []
        for i in range(img_per_folder):
            if result[i] == 0 and i < img_per_folder:
                stat.append(True)
            elif result[i + img_per_folder] == 1 and i > img_per_folder and i < 2 * img_per_folder:
                stat.append(True)
            elif result[i + (2 * img_per_folder)] == 2 and i > 2 * img_per_folder and i < 3 * img_per_folder:
                print("yes")
                stat.append(True)
            else :
                stat.append(False)
        print( sum(stat)/ len(stat) * 100)
            
    
    #fit_save_mlp(300, 50, 50, 0.01, 1000, "Test_remove_me", [5, 5])

    load_predict_mlp(10, 100, 100, "Models/MLP/Test_remove_me_7500_5_5_3.model")
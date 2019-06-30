from dll_load import create_mlp_model, fit_mlp_classification, flatten, predict_mlp_regression, saveModel, loadModel
from load_img import getDataSet
from PIL import Image


if __name__ == "__main__":
    def fit_save():
        img_per_folder = 10
        h = 50
        w = 50
        inputCountPerSample = h * w * 3
        layers = [inputCountPerSample, 60, 10, 3]
        layer_count = 4
        sampleCount = img_per_folder * 3
        
        alpha = 0.04
        epochs = 500

        XTrain, YTrain = getDataSet("../img", img_per_folder, h, w)
        print(YTrain)

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
        file_name = "mlp_dataset"
        for i in layers:
            file_name = file_name + "_" + str(i)
        file_name = file_name + ".model"

        saveModel(W, layers, layer_count, file_name)

    def load_predict():
        h = 50
        w = 50
        layer_count, layers, W = loadModel("mlp_dataset_7500_60_10_3.model")
        XTest = []
        im = Image.open('../img/RTS_Test/RTS_0108.png') 
        imResize = im.resize((h, w), Image.ANTIALIAS)
        imgLoad = imResize.load()
        for x in range(h):
            for y in range(w):
                R,G,B = imgLoad[x, y]
                XTest.append(R)
                XTest.append(G)
                XTest.append(B)
        im.close()
        res = predict_mlp_regression(W, layers, layer_count, h * w * 3, XTest)
        res.pop(0)
        print(res)
    
    #fit_save()
    load_predict()
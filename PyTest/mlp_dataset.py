from dll_load import create_mlp_model, fit_mlp_classification, flatten, predict_mlp_regression, saveModel, loadModel
from load_img import getDataSet
from PIL import Image


if __name__ == "__main__":
    def fit_save():
        img_per_folder = 100
        h = 10
        w = 10
        inputCountPerSample = h * w * 3
        layers = [inputCountPerSample, 32, 32, 64, 64, 3]
        layer_count = 6
        sampleCount = img_per_folder * 3
        
        alpha = 0.04
        epochs = 5000

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
        file_name = "mlp_dataset"
        for i in layers:
            file_name = file_name + "_" + layers[i]
        file_name = file_name + ".model"

        saveModel(W, layers, layer_count, file_name)

    def load_predict():
        h = 10
        w = 10
        layer_count, layers, W = loadModel("mlp_dataset_16_16.model")
        XTest = []
        im = Image.open('../img/FPS/FPS_0000.png') 
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
    
    fit_save()
    load_predict()
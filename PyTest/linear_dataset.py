from dll_load import create_linear_model, fit_classification_rosenblatt_rule, saveLinearModel, loadLinearModel, predict_regression
from load_img import getDataSet
from PIL import Image


if __name__ == "__main__":
    def fit_save():
        img_per_folder = 1
        h = 1000
        w = 1000
        inputCountPerSample = h * w * 3
        sampleCount = img_per_folder * 3
        
        alpha = 0.04
        epochs = 500

        XTrain, Y = getDataSet("../img", img_per_folder, h, w)

        W = create_linear_model(inputCountPerSample)
        YTrain = []

        for i in range(img_per_folder * 3):
            if i < img_per_folder:
                YTrain.append(1)
            else:
                YTrain.append(-1)

        fit_classification_rosenblatt_rule(W, XTrain, sampleCount, inputCountPerSample, YTrain, alpha, epochs)
        file_name = "linear_dataset_" + str(epochs) + ".model"

        saveLinearModel(W, inputCountPerSample, file_name)

    def load_predict():
        h = 1000
        w = 1000
        inputCountPerSample, W = loadLinearModel("mlp_dataset_16_16.model")
        XTest = []
        im = Image.open('../img/RTS/RTS_0000.png') 
        imResize = im.resize((h, w), Image.ANTIALIAS)
        imgLoad = imResize.load()
        for x in range(h):
            for y in range(w):
                R,G,B = imgLoad[x, y]
                XTest.append(R)
                XTest.append(G)
                XTest.append(B)
        im.close()
        res = predict_regression(W, XTest, inputCountPerSample)
        print(res)
    
    fit_save()
    load_predict()
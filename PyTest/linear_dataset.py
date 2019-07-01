from dll_load import create_linear_model, fit_classification_rosenblatt_rule, saveLinearModel, loadLinearModel, predict_regression
from load_img import getDataSet, getDataSetTest
from PIL import Image


if __name__ == "__main__":
    def fit_save():
        img_per_folder = 1200
        h = 100
        w = 100
        inputCountPerSample = h * w * 3
        sampleCount = img_per_folder * 3
        
        alpha = 0.05
        epochs = 500

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
        file_name_FPS = "Models\Linear\linear_dataset_FPS_rendu3_18_30_" + str(img_per_folder) + ".model"
        file_name_MOBA = "Models\Linear\linear_dataset_MOBA_rendu3_18_30_" + str(img_per_folder) + ".model"
        file_name_RTS = "Models\Linear\linear_dataset_RTS_rendu3_18_30_" + str(img_per_folder) + ".model"

        saveLinearModel(W_FPS, inputCountPerSample, file_name_FPS)
        saveLinearModel(W_MOBA, inputCountPerSample, file_name_MOBA)
        saveLinearModel(W_RTS, inputCountPerSample, file_name_RTS)

    def load_predict():
        img_per_folder = 60
        h = 100
        w = 100

        Xpredict = []

        Ypredict_FPS = []
        Ypredict_MOBA = []
        Ypredict_RTS = []

        inputCountPerSample, WFPS = loadLinearModel("Models\Linear\linear_dataset_FPS_rendu3_18_30_1200.model")
        inputCountPerSample, WMOBA = loadLinearModel("Models\Linear\linear_dataset_MOBA_rendu3_18_30_1200.model")
        inputCountPerSample, WRTS = loadLinearModel("Models\Linear\linear_dataset_RTS_rendu3_18_30_1200.model")

        XTest, Y = getDataSetTest("../img", img_per_folder, h, w)
        for img in range(img_per_folder * 3):
            for i in range(h * w * 3):
                Xpredict.append(XTest[img * i])
            Ypredict_FPS.append(predict_regression(WFPS, Xpredict, inputCountPerSample))
            Ypredict_MOBA.append(predict_regression(WMOBA, Xpredict, inputCountPerSample))
            Ypredict_RTS.append(predict_regression(WRTS, Xpredict, inputCountPerSample))
            Xpredict.clear()

        #Predict FPS
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
        


    #fit_save()
    load_predict()
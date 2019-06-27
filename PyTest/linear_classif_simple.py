from dll_load import create_linear_model, fit_classification_rosenblatt_rule, get_Kmeans
from pretty_print import predict_2D
import numpy as np
import random

if __name__ == "__main__":
    sampleCount = 8
    inputCountPerSample = 2
    alpha = 0.02
    epochs = 200
    YTrain = []
    XTrain = [0, 0, 0, 0.5, 0.5, 0, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 1, 1.5, 1.5]
    YTrainKmeans = []
    K = 2

    for val in np.arange(0, sampleCount / 2):
        YTrain.append(-1)
    for val in np.arange(sampleCount / 2, sampleCount):
        YTrain.append(1)

    YTrainKmeans.append(1)
    YTrainKmeans.append(-1)
    W = create_linear_model(inputCountPerSample)
    WKmeans = create_linear_model(inputCountPerSample)

    XTrainKmeans = get_Kmeans(K, XTrain, sampleCount, inputCountPerSample, 2)

    fit_classification_rosenblatt_rule(
        W, XTrain, sampleCount, inputCountPerSample, YTrain, alpha, epochs
    )

    fit_classification_rosenblatt_rule(
        WKmeans, XTrainKmeans, K, inputCountPerSample, YTrainKmeans, alpha, epochs
    )

    predict_2D(W, inputCountPerSample, XTrain, YTrain, 0, 2)
    predict_2D(WKmeans, inputCountPerSample, XTrainKmeans, YTrainKmeans, 0, 2)

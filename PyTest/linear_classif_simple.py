from dll_load import create_linear_model, fit_classification_rosenblatt_rule
from pretty_print import predict_2D
import numpy as np
import random

if __name__ == "__main__":
    sampleCount = 500
    inputCountPerSample = 2
    alpha = 0.02
    epochs = 2000
    YTrain = []
    XTrain = []

    for x in np.arange(0, sampleCount / 2):
            XTrain.append(random.uniform(0, 1) * 0.9)
            XTrain.append(random.uniform(0, 1) * 0.9)

    for x in np.arange(0, sampleCount / 2):
            XTrain.append(random.uniform(1, 2) * 0.9)
            XTrain.append(random.uniform(1, 2) * 0.9)

    for val in np.arange(0, sampleCount / 2):
        YTrain.append(-1)
    for val in np.arange(sampleCount / 2, sampleCount):
        YTrain.append(1)

    W = create_linear_model(inputCountPerSample)

    fit_classification_rosenblatt_rule(W, XTrain, sampleCount, inputCountPerSample, YTrain, alpha, epochs)

    predict_2D(W, inputCountPerSample, XTrain, YTrain, 0, 2)
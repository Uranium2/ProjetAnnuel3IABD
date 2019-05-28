from dll_load import create_mlp_model, fit_mlp_classification, flatten
from pretty_print import predict_2D_mlp_multi
import numpy as np
import random
import matplotlib.pyplot as plt
from collections.abc import Iterable


if __name__ == "__main__":
    layers = [2, 8, 8, 3]
    layer_count = 4
    sampleCount = 1000
    inputCountPerSample = 2
    alpha = 0.01
    epochs = 5000

    X = np.random.random((1000, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in X])
    XTrain = list(flatten(X)) 
    YTrain = list(flatten(Y))


    W = create_mlp_model(layers, layer_count, inputCountPerSample)

    fit_mlp_classification(W, XTrain, YTrain, layers, layer_count, sampleCount, inputCountPerSample, alpha, epochs)

    predict_2D_mlp_multi(W, layers, layer_count, inputCountPerSample, XTrain, Y)

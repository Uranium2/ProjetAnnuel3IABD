from dll_load import create_mlp_model, fit_mlp_classification, flatten
from pretty_print import predict_2D_mlp
import numpy as np
import random

if __name__ == "__main__":
    layers = [2, 4 , 1]
    layer_count = 3
    sampleCount = 500
    inputCountPerSample = 2
    alpha = 0.01
    epochs = 10000

    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X])

    XTrain = list(flatten(X)) 
    YTrain = list(flatten(Y))


    W = create_mlp_model(layers, layer_count)

    fit_mlp_classification(W, XTrain, YTrain, layers, layer_count, sampleCount, inputCountPerSample, alpha, epochs)

    predict_2D_mlp(W, layers, layer_count, inputCountPerSample, XTrain, YTrain, -1 , 1)

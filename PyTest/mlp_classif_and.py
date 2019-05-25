from dll_load import create_mlp_model, fit_mlp_classification
from pretty_print import predict_mlp_2D_AND

if __name__ == "__main__":
    layers = [2, 1]
    layer_count = 2
    sampleCount = 4
    inputCountPerSample = 2
    alpha = 0.02
    epochs = 2000
    YTrain = [-1, -1, -1, 1]
    XTrain = [0, 0, 0, 1, 1, 0, 1, 1]


    W = create_mlp_model(layers, layer_count, inputCountPerSample)

    fit_mlp_classification(W, XTrain, YTrain, layers, layer_count, sampleCount, inputCountPerSample, alpha, epochs)

    #predict_mlp_2D_AND(W, layers, layer_count, inputCountPerSample)

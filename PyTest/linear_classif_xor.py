from dll_load import create_linear_model, fit_classification_rosenblatt_rule
from pretty_print import predict_2D_XOR

if __name__ == "__main__":
    sampleCount = 4
    inputCountPerSample = 2
    alpha = 0.02
    epochs = 1000
    YTrain = [1, -1, -1, 1]
    XTrain = [0, 0, 0, 1, 1, 0, 1, 1]

    W = create_linear_model(inputCountPerSample)

    fit_classification_rosenblatt_rule(W, XTrain, sampleCount, inputCountPerSample, YTrain, alpha, epochs)

    predict_2D_XOR(W, inputCountPerSample)
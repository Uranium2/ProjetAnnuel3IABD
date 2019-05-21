from dll_load import myDll, create_linear_model, fit_regression
from pretty_print import predict_2D_reg

if __name__ == "__main__":
    sampleCount = 3
    inputCountPerSample = 1
    XTrain = [1, 2, 3]
    YTrain = [2, 3, 2.5]

    W = fit_regression(XTrain, sampleCount, inputCountPerSample, YTrain)

    predict_2D_reg(W, inputCountPerSample, XTrain, YTrain)
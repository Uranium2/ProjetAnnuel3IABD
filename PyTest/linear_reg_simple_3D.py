from dll_load import create_linear_model, fit_regression
from pretty_print import predict_3D_reg

if __name__ == "__main__":
    sampleCount = 3
    inputCountPerSample = 2
    XTrain = [1, 1, 2, 2, 3 , 1]
    YTrain = [2, 3, 2.5]

    W = fit_regression(XTrain, sampleCount, inputCountPerSample, YTrain)

    predict_3D_reg(W, inputCountPerSample, XTrain, YTrain)
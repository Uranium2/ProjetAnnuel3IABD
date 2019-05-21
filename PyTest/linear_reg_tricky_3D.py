from dll_load import myDll, create_linear_model, fit_regression
from pretty_print import predict_3D_reg

if __name__ == "__main__":
    sampleCount = 3
    inputCountPerSample = 2
    XTrain = [1, 1, 2, 2, 3 , 3]
    YTrain = [1, 2, 3]

    W = fit_regression(XTrain, sampleCount, inputCountPerSample, YTrain)

    predict_3D_reg(W, inputCountPerSample, XTrain, YTrain)
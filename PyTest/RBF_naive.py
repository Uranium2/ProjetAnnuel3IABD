from dll_load import fit_reg_RBF_naive

from pretty_print import predict_2D_RBF

gamma = 500
sampleCount = 5
inputCountPerSample = 2
XTrain = [0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 5.0, 5.0, 2.5, 2.5]
YTrain = [-1.0, -1.0, -1.0, -1.0, 1.0]


W = fit_reg_RBF_naive(XTrain, gamma, YTrain, sampleCount, inputCountPerSample)

predict_2D_RBF(W, XTrain, YTrain, inputCountPerSample, gamma, sampleCount, 0, 5)

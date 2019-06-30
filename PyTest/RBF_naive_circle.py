from dll_load import fit_reg_RBF_naive, flatten

from pretty_print import predict_2D_RBF

from sklearn.datasets import make_circles

# generate 2d classification dataset
X, Y = make_circles(n_samples=100, noise=0.06)
XTrain = list(flatten(X))
YTrain0 = list(flatten(Y))
YTrain = []
for i in YTrain0:
    if i == 0:
        YTrain.append(-1)
    else:
        YTrain.append(1)

XTrain.append(0)
XTrain.append(0)

YTrain.append(1)

gamma = 500
sampleCount = 101
inputCountPerSample = 2

W = fit_reg_RBF_naive(XTrain, gamma, YTrain, sampleCount, inputCountPerSample)

predict_2D_RBF(W, XTrain, YTrain, inputCountPerSample, gamma, sampleCount, -1, 1)

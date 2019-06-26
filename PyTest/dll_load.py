from ctypes import *
from ctypes.wintypes import *
import ctypes as ct
import os
from collections.abc import Iterable

dll_name = "..\\src\\x64\\Debug\\2019-3A-IBD-MLDLL.dll"
dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
myDll = CDLL(dllabspath)

# create_linear_model
myDll.create_linear_model.argtypes = [ct.c_int]

# fit_classification_rosenblatt_rule
myDll.create_linear_model.restype = ct.c_void_p
myDll.fit_classification_rosenblatt_rule.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
    ct.c_double,
    ct.c_int,
]

# predict_classification_rosenblatt
myDll.predict_classification_rosenblatt.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
myDll.predict_classification_rosenblatt.restype = ct.c_double

# fit_regression
myDll.fit_regression.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p]
myDll.fit_regression.restype = ct.c_void_p

# predict_regression
myDll.predict_regression.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
myDll.predict_regression.restype = ct.c_double

# create_mlp_model
myDll.create_mlp_model.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
myDll.create_mlp_model.restype = ct.c_void_p

# fit_mlp_classification
myDll.fit_mlp_classification.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_double,
    ct.c_int,
]

# fit_mlp_regression
myDll.fit_mlp_regression.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_double,
    ct.c_int,
]


# predict_mlp_classification
myDll.predict_mlp_classification.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
]
myDll.predict_mlp_classification.restype = POINTER(ct.c_double)

# predict_mlp_regression
myDll.predict_mlp_regression.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_void_p,
]
myDll.predict_mlp_regression.restype = POINTER(ct.c_double)

# fit_reg_RBF_naive
myDll.fit_reg_RBF_naive.argtypes = [
    ct.c_void_p,
    ct.c_double,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
]
myDll.fit_reg_RBF_naive.restype = ct.c_void_p

# predict_reg_RBF_naive
myDll.predict_reg_RBF_naive.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_int,
    ct.c_double,
    ct.c_int,
]
myDll.predict_reg_RBF_naive.restype = ct.c_double

# get_Kmeans
myDll.get_Kmeans.argtypes = [
    ct.c_int,
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_int,
]
myDll.get_Kmeans.restype =  POINTER(ct.c_double)

def create_linear_model(pyInputCountPerSample):
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    return myDll.create_linear_model(inputCountPerSample)


def fit_classification_rosenblatt_rule(
    W, pyXTrain, pySampleCount, pyInputCountPerSample, pyYTrain, pyAlpha, pyEpochs
):
    sampleCount = ct.c_int(pySampleCount)
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    alpha = ct.c_double(pyAlpha)
    epochs = ct.c_int(pyEpochs)
    YTrain = (ct.c_double * len(pyYTrain))(*pyYTrain)
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    myDll.fit_classification_rosenblatt_rule(
        W, XTrain, sampleCount, inputCountPerSample, YTrain, alpha, epochs
    )


def predict_classification_rosenblatt(W, pyX, pyInputCountPerSample):
    X = (ct.c_double * len(pyX))(*pyX)
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    return myDll.predict_classification_rosenblatt(W, X, inputCountPerSample)


def fit_regression(pyXTrain, pySampleCount, pyInputCountPerSample, pyYTrain):
    sampleCount = ct.c_int(pySampleCount)
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    YTrain = (ct.c_double * len(pyYTrain))(*pyYTrain)
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    return myDll.fit_regression(XTrain, sampleCount, inputCountPerSample, YTrain)


def predict_regression(W, pyX, pyInputCountPerSample):
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    X = (ct.c_double * len(pyX))(*pyX)
    return myDll.predict_regression(W, X, inputCountPerSample)


def create_mlp_model(pyLayers, pyLayer_count, pyInputCountPerSample):
    layers = (ct.c_int * len(pyLayers))(*pyLayers)
    layer_count = ct.c_int(pyLayer_count)
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    return myDll.create_mlp_model(layers, layer_count, inputCountPerSample)


def fit_mlp_classification(
    W,
    pyXTrain,
    pyYTrain,
    pyLayers,
    pyLayer_count,
    pySampleCount,
    pyInputCountPerSample,
    pyAlpha,
    pyEpochs,
):
    layers = (ct.c_int * len(pyLayers))(*pyLayers)
    layer_count = ct.c_int(pyLayer_count)
    sampleCount = ct.c_int(pySampleCount)
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    alpha = ct.c_double(pyAlpha)
    epochs = ct.c_int(pyEpochs)
    YTrain = (ct.c_int * len(pyYTrain))(*pyYTrain)
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    myDll.fit_mlp_classification(
        W,
        XTrain,
        YTrain,
        layers,
        layer_count,
        sampleCount,
        inputCountPerSample,
        alpha,
        epochs,
    )


def predict_mlp_classification(W, pyLayers, pyLayer_count, pyInputCountPerSample, pyX):
    layers = (ct.c_int * len(pyLayers))(*pyLayers)
    layer_count = ct.c_int(pyLayer_count)
    X = (ct.c_double * len(pyX))(*pyX)
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    res = myDll.predict_mlp_classification(
        W, layers, layer_count, inputCountPerSample, X
    )
    l = [res[i] for i in range(pyLayers[-1] + 1)]
    return l


def fit_mlp_regression(
    W,
    pyXTrain,
    pyYTrain,
    pyLayers,
    pyLayer_count,
    pySampleCount,
    pyInputCountPerSample,
    pyAlpha,
    pyEpochs,
):
    layers = (ct.c_int * len(pyLayers))(*pyLayers)
    layer_count = ct.c_int(pyLayer_count)
    sampleCount = ct.c_int(pySampleCount)
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    alpha = ct.c_double(pyAlpha)
    epochs = ct.c_int(pyEpochs)
    YTrain = (ct.c_int * len(pyYTrain))(*pyYTrain)
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    myDll.fit_mlp_regression(
        W,
        XTrain,
        YTrain,
        layers,
        layer_count,
        sampleCount,
        inputCountPerSample,
        alpha,
        epochs,
    )


def predict_mlp_regression(W, pyLayers, pyLayer_count, pyInputCountPerSample, pyX):
    layers = (ct.c_int * len(pyLayers))(*pyLayers)
    layer_count = ct.c_int(pyLayer_count)
    X = (ct.c_double * len(pyX))(*pyX)
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    res = myDll.predict_mlp_regression(W, layers, layer_count, inputCountPerSample, X)
    l = [res[i] for i in range(pyLayers[-1] + 1)]
    return l


def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def fit_reg_RBF_naive(
    pyXTrain, pyGamma, pyYTrain, pySampleCount, pyInputCountPerSample
):
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    YTrain = (ct.c_double * len(pyYTrain))(*pyYTrain)
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    sampleCount = ct.c_int(pySampleCount)
    gamma = ct.c_double(pyGamma)
    return myDll.fit_reg_RBF_naive(
        XTrain, gamma, YTrain, sampleCount, inputCountPerSample
    )


def predict_reg_RBF_naive(
    W, pyXTrain, pyXpredict, pyInputCountPerSample, pyGamma, pySampleCount
):
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    sampleCount = ct.c_int(pySampleCount)
    gamma = ct.c_double(pyGamma)
    Xpredict = (ct.c_double * len(pyXpredict))(*pyXpredict)
    return myDll.predict_reg_RBF_naive(
        W, XTrain, Xpredict, inputCountPerSample, gamma, sampleCount
    )

def get_Kmeans(pyK, pyXTrain, pySampleCount, pyInputCountPerSample, pyEpochs):
    K = ct.c_int(pyK)
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    sampleCount = ct.c_int(pySampleCount)
    epochs = ct.c_int(pyEpochs)
    kmeansC =  myDll.get_Kmeans(K, XTrain, sampleCount, inputCountPerSample, epochs)
    kmeans = [kmeansC[i] for i in range(pyK * pyInputCountPerSample)]
    return kmeans

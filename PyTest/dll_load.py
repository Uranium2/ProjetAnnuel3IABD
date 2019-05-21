from ctypes import *
from ctypes.wintypes import *
import ctypes as ct
import os


dll_name = "..\\src\\x64\\Debug\\2019-3A-IBD-MLDLL.dll"
dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
myDll = CDLL(dllabspath)

# create_linear_model
myDll.create_linear_model.argtypes = [ct.c_int]

# fit_classification_rosenblatt_rule
myDll.create_linear_model.restype = ct.POINTER(ct.c_double * 3)
myDll.fit_classification_rosenblatt_rule.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p, ct.c_double, ct.c_int]

# predict_classification_rosenblatt
myDll.predict_classification_rosenblatt.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
myDll.predict_classification_rosenblatt.restype = ct.c_double

# fit_regression
myDll.fit_regression.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p]

# predict_regression
myDll.predict_regression.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
myDll.predict_regression.restype = ct.c_double

# create_mlp_model
myDll.create_mlp_model.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
myDll.create_mlp_model.restype = ct.c_void_p


def create_linear_model(pyInputCountPerSample):
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    return myDll.create_linear_model(inputCountPerSample)

def fit_classification_rosenblatt_rule(W, pyXTrain, pySampleCount, pyInputCountPerSample, pyYTrain, pyAlpha, pyEpochs):
    sampleCount = ct.c_int(pySampleCount)
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    alpha = ct.c_double(pyAlpha)
    epochs = ct.c_int(pyEpochs)
    YTrain = (ct.c_double * len(pyYTrain))(*pyYTrain)
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)
    myDll.fit_classification_rosenblatt_rule(W, XTrain, sampleCount, inputCountPerSample, YTrain, alpha, epochs)

def predict_classification_rosenblatt(W, pyX, pyInputCountPerSample):
    X = (ct.c_double * len(pyX))(*pyX)
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    return myDll.predict_classification_rosenblatt(W, X, inputCountPerSample)

def predict_regression(W, pyX, pyInputCountPerSample):
    inputCountPerSample = ct.c_int(pyInputCountPerSample)
    X = (ct.c_double * len(pyX))(*pyX)
    return myDll.predict_regression(W,  X, inputCountPerSample)
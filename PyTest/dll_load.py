from ctypes import *
from ctypes.wintypes import *
import ctypes as ct
import os
import matplotlib.pyplot as plt

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

def predict_2D(W, inputCountPerSample):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for x in range(0, 200):
        for y in range(0, 200):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res = predict_regression(W, dot, inputCountPerSample)
            if ( res > 0):
                x1.append(x / 100)
                y1.append(y / 100)
            else:
                x2.append(x / 100)
                y2.append(y / 100)

    plt.scatter(x1, y1, c = 'green')
    plt.scatter(x2, y2, c = 'red')
    plt.show()


def predict_2D_OR(W, inputCountPerSample):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for x in range(0, 100):
        for y in range(0, 100):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res = predict_regression(W, dot, inputCountPerSample)
            if ( res > 0):
                x1.append(x / 100)
                y1.append(y / 100)
            else:
                x2.append(x / 100)
                y2.append(y / 100)


    x3 = [0]
    y3 = [0]
    x4 = [0, 1, 1]
    y4 = [1, 0, 1]
    plt.scatter(x1, y1, c = 'green')
    plt.scatter(x2, y2, c = 'red')
    plt.scatter(x3, y3, c = 'yellow')
    plt.scatter(x4, y4, c = 'magenta')
    plt.show()

def predict_2D_XOR(W, inputCountPerSample):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for x in range(0, 100):
        for y in range(0, 100):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res = predict_regression(W, dot, inputCountPerSample)
            if ( res > 0):
                x1.append(x / 100)
                y1.append(y / 100)
            else:
                x2.append(x / 100)
                y2.append(y / 100)


    x3 = [0, 1]
    y3 = [0, 1]
    x4 = [0, 1]
    y4 = [1, 0]
    plt.scatter(x1, y1, c = 'green')
    plt.scatter(x2, y2, c = 'red')
    plt.scatter(x3, y3, c = 'yellow')
    plt.scatter(x4, y4, c = 'magenta')
    plt.show()

def predict_2D_AND(W, inputCountPerSample):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for x in range(0, 100):
        for y in range(0, 100):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res = predict_regression(W, dot, inputCountPerSample)
            if ( res > 0):
                x1.append(x / 100)
                y1.append(y / 100)
            else:
                x2.append(x / 100)
                y2.append(y / 100)


    x3 = [0, 0, 1]
    y3 = [0, 1, 0]
    x4 = [1]
    y4 = [1]
    plt.scatter(x1, y1, c = 'green')
    plt.scatter(x2, y2, c = 'red')
    plt.scatter(x3, y3, c = 'yellow')
    plt.scatter(x4, y4, c = 'magenta')
    plt.show()


def predict_2D_3Class(W1, W2, W3, inputCountPerSample,x3, y3, x4, y4, x5, y5):
    x11 = []
    y11 = []
    x12 = []
    y12 = []
    x13 = []
    y13 = []

    for x in range(0, 200):
        for y in range(0, 200):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res1 = predict_regression(W1, dot, inputCountPerSample)
            res2 = predict_regression(W2, dot, inputCountPerSample)
            res3 = predict_regression(W3, dot, inputCountPerSample)
            l = []
            l.append(res1)
            l.append(res2)
            l.append(res3)
            if ( res1 == max(l)):
                x11.append(x / 100)
                y11.append(y / 100)
            if ( res2 == max(l)):
                x12.append(x / 100)
                y12.append(y / 100)
            if ( res3 == max(l)):
                x13.append(x / 100)
                y13.append(y / 100)



    plt.scatter(x11, y11, c = 'green')
    plt.scatter(x12, y12, c = 'red')
    plt.scatter(x13, y13, c = 'cyan')
    
    plt.scatter(x3, y3, c = 'yellow')
    plt.scatter(x4, y4, c = 'magenta')
    plt.scatter(x5, y5, c = 'blue')
    plt.show()

def predict_2D_3Class_individual(W, inputCountPerSample,x3, y3, x4, y4, x5, y5):
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for x in range(0, 200):
        for y in range(0, 200):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res = predict_regression(W, dot, inputCountPerSample)
            if ( res > 0):
                x1.append(x / 100)
                y1.append(y / 100)
            else:
                x2.append(x / 100)
                y2.append(y / 100)



    plt.scatter(x1, y1, c = 'green')
    plt.scatter(x2, y2, c = 'red')
    
    plt.scatter(x3, y3, c = 'yellow')
    plt.scatter(x4, y4, c = 'magenta')
    plt.scatter(x5, y5, c = 'blue')
    plt.show()
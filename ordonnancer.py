from ctypes import *
from ctypes.wintypes import *
import ctypes as ct

if __name__ == "__main__":
    sampleCount = ct.c_int(4)
    inputCountPerSample = ct.c_int(2)
    alpha = ct.c_double(0.01)
    epochs = ct.c_int(200)
    pyYTrain = [-1, 1, 1, 1]
    YTrain = (ct.c_double * len(pyYTrain))(*pyYTrain)
    pyXTrain = [0, 0, 0, 1, 1, 0, 1, 1]
    XTrain = (ct.c_double * len(pyXTrain))(*pyXTrain)


    myDll = CDLL("C:/Users/Tavernier/Documents/ProjetAnnuel3IABD/src/x64/Debug/2019-3A-IBD-MLDLL.dll")
    myDll.create_linear_model.argtypes = [ct.c_int]
    myDll.create_linear_model.restype = ct.POINTER(ct.c_double * 3)
    myDll.fit_classification_rosenblatt_rule.argtypes = [ct.POINTER(ct.c_double * 3), ct.POINTER(ct.c_double * len(pyXTrain)), ct.c_int, ct.c_int, ct.POINTER(ct.c_double), ct.c_double, ct.c_int]
    

    W = myDll.create_linear_model(inputCountPerSample)
    myDll.fit_classification_rosenblatt_rule(W, XTrain, sampleCount, inputCountPerSample, YTrain, alpha, epochs)

    pyX1 = [0, 0]
    X1 = (ct.c_double * len(pyX1))(*pyX1)
    pyX2 = [0, 1]
    X2 = (ct.c_double * len(pyX2))(*pyX2)
    pyX3 = [1, 0]
    X3 = (ct.c_double * len(pyX3))(*pyX3)
    pyX4 = [1, 1]
    X4 = (ct.c_double * len(pyX4))(*pyX4)

    myDll.predict_regression.argtypes = [ct.POINTER(ct.c_double * 3), ct.POINTER(ct.c_double * len(pyX3)), ct.c_int]
    myDll.predict_regression.restype = ct.c_double

    print(myDll.predict_regression(W, X1, inputCountPerSample))
    print(myDll.predict_regression(W, X2, inputCountPerSample))
    print(myDll.predict_regression(W, X3, inputCountPerSample))
    print(myDll.predict_regression(W, X4, inputCountPerSample))

    layers_count = ct.c_int(2)
    inputCountPerSample = ct.c_int(2)

    myDll.create_mlp_model.argtypes = [ct.POINTER(ct.c_int), ct.c_int, ct.c_int]
    myDll.create_mlp_model.restype = ct.POINTER(ct.POINTER(ct.POINTER(ct.c_double)))
    #myDll.printArrayPython3D.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_int * 2), ct.c_int, ct.c_int]
    #myDll.printArrayPython3D.restype = c_void_p

    pyLayers = [2, 1]
    Layers = (ct.c_int * len(pyLayers))(*pyLayers)
    
    W =  myDll.create_mlp_model(Layers, layers_count, inputCountPerSample)


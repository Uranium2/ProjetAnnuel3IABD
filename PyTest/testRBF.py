from ctypes import *
from ctypes.wintypes import *
import ctypes as ct
import os
from collections.abc import Iterable

dll_name = "..\\src\\x64\\Debug\\2019-3A-IBD-MLDLL.dll"
dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
myDll = CDLL(dllabspath)

#fit_reg_RBF_naive
myDll.fit_reg_RBF_naive.argtypes = [ct.c_void_p, ct.c_double,  ct.c_void_p, ct.c_int, ct.c_int]
myDll.fit_reg_RBF_naive.restypes = ct.c_void_p

#predict_reg_RBF_naive
myDll.predict_reg_RBF_naive.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_double, ct.c_int]
myDll.predict_reg_RBF_naive.restypes = ct.c_double
gamma = 50
sampleCount = 2
inputCountPerSample = 3
XTrain = [1.0, 1.0, 1.0, 3.0, 3.0, 3.0]
YTrain = [-1.0, 1.0]

XTrain = (ct.c_double * len(XTrain))(*XTrain)
YTrain = (ct.c_double * len(YTrain))(*YTrain)
inputCountPerSample = ct.c_int(inputCountPerSample)
sampleCount = ct.c_int(sampleCount)
gamma = ct.c_double(gamma)

Xpredict = [1.0, 1.0, 1.0]
Xpredict = (ct.c_double * len(Xpredict))(*Xpredict)

W = myDll.fit_reg_RBF_naive(XTrain, gamma, YTrain, sampleCount, inputCountPerSample)
print(id(W))
myDll.predict_reg_RBF_naive(W, XTrain, Xpredict,inputCountPerSample, gamma, sampleCount)
from ctypes import *
from ctypes.wintypes import *
import ctypes as ct
import os
from collections.abc import Iterable

dll_name = "..\\src\\x64\\Debug\\2019-3A-IBD-MLDLL.dll"
dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
myDll = CDLL(dllabspath)

#Get distance

myDll.get_distance.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
myDll.get_distance.restypes = ct.c_double

inputCountPerSample = 3

XPredict = [
    1, 1, 1
]

Xn = [3, 3, 3]

inputCountPerSample = ct.c_int(inputCountPerSample)
XPredict = (ct.c_double * len(XPredict))(*XPredict)
Xn = (ct.c_double * len(Xn))(*Xn)

myDll.get_distance(XPredict, Xn, inputCountPerSample)
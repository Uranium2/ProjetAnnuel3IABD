from ctypes import *
from ctypes.wintypes import *
import ctypes

if __name__ == "__main__":
    myDll = CDLL("C:/Users/matt_/Documents/Cours IBD/Projet Annuel/ProjetAnnuel3IABD/src/x64/Debug/2019-3A-IBD-MLDLL.dll")
    myDll.create_linear_model.argtypes = [c_int32]
    myDll.create_linear_model.restype = ctypes.POINTER(ctypes.c_double * 10)

    #myDll.test_python_array.argtypes[c_void_p]

    ArrayType = ctypes.c_double*10
    W = myDll.create_linear_model(10)
    myDll.test_python_array(W, 10)

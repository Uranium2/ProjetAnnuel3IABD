from ctypes import *
from ctypes.wintypes import *
import ctypes as ct

if __name__ == "__main__":

    myDll = CDLL("C:\\Users\matt_\\Documents\\Cours IBD\\Projet Annuel\\ProjetAnnuel3IABD\\src\\x64\\Debug\\2019-3A-IBD-MLDLL.dll")
    #myDll.HelloWorld()

    #print ("Hello python")

    layers_count = ct.c_int(2)
    inputCountPerSample = ct.c_int(2)

    myDll.create_mlp_model.argtypes = [ct.POINTER(ct.c_int), ct.c_int, ct.c_int]
    myDll.create_mlp_model.restype = ct.POINTER(ct.POINTER(ct.POINTER(ct.c_double)))
    #myDll.printArrayPython3D.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_int * 2), ct.c_int, ct.c_int]
    #myDll.printArrayPython3D.restype = c_void_p

    pyLayers = [2, 1]
    Layers = (ct.c_int * len(pyLayers))(*pyLayers)
    
    W =  myDll.create_mlp_model(Layers, layers_count, inputCountPerSample)
import numpy as np
from load_img import *
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import h5py



def mlp_model(img_per_folder, height, width, layers, imageToPredict, batch_size=1, epochs=500, W_MLP=None):

    inputCountPerSample = height * width * 3

    #retrieve images
    XTrain, Y = getDataSet("../img", img_per_folder, height, width, False)
    XTrain = np.array(XTrain)
    Y = np.array(Y)
    XTrain = XTrain.reshape(3 * img_per_folder, inputCountPerSample)

    
    if W_MLP == None:

        #create model
        W_MLP = Sequential()
        for neuron in layers[1:1]:
            W_MLP.add(Dense(neuron, activation='tanh', input_dim=XTrain.shape[1]))

        #compile model
        W_MLP.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
        
        #fit model
        W_MLP.fit(XTrain, Y, batch_size=batch_size, verbose=1, epochs=epochs)

        #save model
        W_MLP.save("models/MLP_Model")

    else:
        W_MLP = W_MLP

    #Evaluate model
    Xpredict = []

    size = inputCountPerSample / 3
    size = int(math.sqrt(size))

    im = Image.open(imageToPredict)
    im = im.convert("RGB")
    imResize = im.resize((size, size), Image.ANTIALIAS)
    imgLoad = imResize.load()

    for x in range(size):
        for y in range(size):
            R,G,B = imgLoad[x, y]
            Xpredict.append(R / 255)
            Xpredict.append(G / 255)
            Xpredict.append(B / 255)
    im.close()

    Xpredict = np.array(Xpredict)
    Xpredict = Xpredict.reshape(1, inputCountPerSample)

    return W_MLP.predict(Xpredict)

print(mlp_model(3, 20, 20, [1, 1, 1], "../img/RTS_Test/RTS_0140.png"))
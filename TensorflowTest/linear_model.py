import numpy as np
from load_img import *
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import h5py


def load_linear_model(model_path):
    model = load_model(model_path)
    return model


def linear_keras(filename, img_per_folder, height, width, batch_size=1, epochs=100):

    inputCountPerSample = height * width * 3

    # retrieve images
    XTrain, Y = getDataSet("../img", img_per_folder, height, width, False)
    XTrain = np.array(XTrain)
    XTrain = XTrain.reshape(3 * img_per_folder, inputCountPerSample)
    # print(XTrain.shape)

    # create models
    W_FPS = Sequential()
    W_FPS.add(Dense(1, activation="tanh", input_dim=XTrain.shape[1]))

    W_MOBA = Sequential()
    W_MOBA.add(Dense(1, activation="tanh", input_dim=XTrain.shape[1]))

    W_RTS = Sequential()
    W_RTS.add(Dense(1, activation="tanh", input_dim=XTrain.shape[1]))

    # compile models
    W_FPS.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])
    W_MOBA.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])
    W_RTS.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])

    YTrain_FPS = np.array([])
    YTrain_MOBA = np.array([])
    YTrain_RTS = np.array([])

    for i in range(img_per_folder * 3):
        if i < img_per_folder:
            YTrain_FPS = np.append(YTrain_FPS, 1)
        else:
            YTrain_FPS = np.append(YTrain_FPS, -1)
    for i in range(img_per_folder * 3):
        if i >= img_per_folder and i < 2 * img_per_folder:
            YTrain_MOBA = np.append(YTrain_MOBA, 1)
        else:
            YTrain_MOBA = np.append(YTrain_MOBA, -1)
    for i in range(img_per_folder * 3):
        if i >= 2 * img_per_folder and i < 3 * img_per_folder:
            YTrain_RTS = np.append(YTrain_RTS, 1)
        else:
            YTrain_RTS = np.append(YTrain_RTS, -1)

    # fit models
    W_FPS.fit(XTrain, YTrain_FPS, batch_size=batch_size, verbose=1, epochs=epochs)
    W_MOBA.fit(XTrain, YTrain_MOBA, batch_size=batch_size, verbose=1, epochs=epochs)
    W_RTS.fit(XTrain, YTrain_RTS, batch_size=batch_size, verbose=1, epochs=epochs)

    # save models
    W_FPS.save("models/" + filename + "_" + str(inputCountPerSample) + "_FPS.model")
    W_MOBA.save("models/" + filename + "_" + str(inputCountPerSample) + "_MOBA.model")
    W_RTS.save("models/" + filename + "_" + str(inputCountPerSample) + "_RTS.model")


def get_stats_linear_tf(img_per_folder, pathFPS, pathMOBA, pathRTS, isValidation):
	FPS_model = load_linear_model("models/" + pathFPS)
	MOBA_model = load_linear_model("models/" + pathMOBA)
	RTS_model = load_linear_model("models/" + pathRTS)

	inputCountPerSample = int(pathFPS.split("_")[1])
	size = inputCountPerSample / 3
	size = int(math.sqrt(size))

	files = getImgPath("../img", img_per_folder, size, size, isValidation)
	result = []
	for img in files:
		fps, moba, rts, index = predict_linear_tf(FPS_model, MOBA_model, RTS_model, img, inputCountPerSample)
		result.append(index)

	stat = []
	for i in range(img_per_folder * 3):
		if result[i] == 0 and i < img_per_folder:
			stat.append(True)
		elif result[i] == 1 and i >= img_per_folder and i < 2 * img_per_folder:
			stat.append(True)
		elif result[i] == 2 and i >= 2 * img_per_folder and i < 3 * img_per_folder:
			stat.append(True)
		else :
			stat.append(False)
	print(sum(stat)/ len(stat) * 100)
	return (sum(stat)/ len(stat) * 100)
	

def predict_linear_tf(FPS_model, MOBA_model, RTS_model, imageToPredict, inputCountPerSample):
	# Evaluate models
	Xpredict = []

	size = inputCountPerSample / 3
	size = int(math.sqrt(size))

	im = Image.open(imageToPredict)
	im = im.convert("RGB")
	imResize = im.resize((size, size), Image.ANTIALIAS)
	imgLoad = imResize.load()

	for x in range(size):
		for y in range(size):
			R, G, B = imgLoad[x, y]
			Xpredict.append(R / 255)
			Xpredict.append(G / 255)
			Xpredict.append(B / 255)
	im.close()

	Xpredict = np.array(Xpredict)
	Xpredict = Xpredict.reshape(1, inputCountPerSample)
	fps = FPS_model.predict(Xpredict)[0][0]
	moba = MOBA_model.predict(Xpredict)[0][0]
	rts = RTS_model.predict(Xpredict)[0][0]
	index = 0
	if fps > moba and fps > rts:
		index = 0
	if moba > fps and moba > rts:
		index = 1
	if rts > moba and rts > fps:
		index = 2
	print(fps, moba, rts, index)
	return ( fps,
			moba,
			rts,
			index
	)


#linear_keras("hello", 50, 20, 20)

#predict_linear_tf("hello_1200_FPS.model", "hello_1200_MOBA.model", "hello_1200_RTS.model", "..\\img\\RTS_Validation\\RTS_Validation_0001.png")

get_stats_linear_tf(600, "hello_1200_FPS.model", "hello_1200_MOBA.model", "hello_1200_RTS.model", False)
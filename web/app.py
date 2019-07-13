from flask import Flask, request, render_template, send_from_directory
import os
import sys
from flask_material import Material
from linear_dataset import fit_save_classif, load_predict_classif, linear_keras, web_predict_linear_tf
from mlp_dataset import fit_save_mlp, load_predict_mlp
from os import listdir
from os.path import isfile, join
import csv
import pandas as pd
import uuid



app = Flask(__name__)
Material(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def buildModelFolders():
    pathModelLinear = APP_ROOT + "\Models\\Linear\\"
    pathModelLinearTF = APP_ROOT + "\Models\\LinearTF\\"
    pathModelMLP = APP_ROOT + "\Models\\MLP\\"
    pathModelRBF = APP_ROOT + "\Models\\RBF\\"
    if not os.path.exists(pathModelLinear):
        os.makedirs(pathModelLinear)
    if not os.path.exists(pathModelLinearTF):
        os.makedirs(pathModelLinearTF)
    if not os.path.exists(pathModelMLP):
        os.makedirs(pathModelMLP)
    if not os.path.exists(pathModelRBF):
        os.makedirs(pathModelRBF)
    

def getOldPredict():
    res = []
    with open('static/prediction.csv') as csvfile:
        rows = csv.reader(csvfile)
        res.append(list(zip(*rows)))

    res = list(map(list, zip(*res[0])))

    return res

@app.route('/')
def index():
    buildModelFolders()
    myLinearPath = "Models/Linear/"
    myLinearTFPath = "Models/LinearTF/"
    myMlpPath = "Models/MLP/"
    myRBFPath = "Models/RBF/"
    
    res = getOldPredict()
    linearFiles = [f for f in listdir(
        myLinearPath) if isfile(join(myLinearPath, f))]
    linearTFFiles = [f for f in listdir(
        myLinearTFPath) if isfile(join(myLinearTFPath, f))]
    mlpFiles = [f for f in listdir(myMlpPath) if isfile(join(myMlpPath, f))]
    RBFFiles = [f for f in listdir(myRBFPath) if isfile(join(myRBFPath, f))]
    return render_template('index.html', linear=linearFiles, lineartf=linearTFFiles, mlp=mlpFiles, rbf=RBFFiles, oldpredict=res)


@app.route("/upload", methods=['POST'])
def upload():
    
    target = os.path.join(APP_ROOT, 'static/images')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        unique_filename = str(uuid.uuid4()) + ".PNG"
        dest = "/".join([target, unique_filename])
        print(dest)
        file.save(dest)
    imgPath = "./static/images/" + unique_filename


    stat = []
    model = request.form['model']
    if model == "Linear Model":
        fps = request.form['file1']
        moba = request.form['file2']
        rts = request.form['file3']
        print("LINEAR PREDICT")
        Ypredict_FPS, Ypredict_MOBA, Ypredict_RTS, result = load_predict_classif("Models/Linear/" + fps,
                                                                             "Models/Linear/" + moba,
                                                                             "Models/Linear/" + rts,
                                                                             dest)
        stat.append(Ypredict_FPS)
        stat.append(Ypredict_MOBA)
        stat.append(Ypredict_RTS)
        print(stat)
    elif model == "Linear Model Tensorflow":
        fps = request.form['filestf1']
        moba = request.form['filestf2']
        rts = request.form['filestf2']
        Ypredict_FPS, Ypredict_MOBA, Ypredict_RTS, result = web_predict_linear_tf(fps, moba, rts, dest, 7500)
        print(Ypredict_FPS)
        print(Ypredict_MOBA)
        print(Ypredict_RTS)
        print(result)
        stat.append([Ypredict_FPS])
        stat.append([Ypredict_MOBA])
        stat.append([Ypredict_RTS])
        print(stat)
    elif model == "Multilayer perceptron":
        print("MLP PREDICT")
        mlp = request.form['file4']
        Ypredict, result = load_predict_mlp("Models/MLP/" + mlp, dest)
        for y in Ypredict[0]:
            stat.append([y])
        print(stat)
    elif model == "RBF":
        print("RBF PREDICT")

    res = getOldPredict()
    if result == 0:
        return render_template("result.html", data=stat, res="I think this game is a FPS", oldpredict=res, user_image=imgPath)
    if result == 1:
        return render_template("result.html", data=stat, res="I think this game is a MOBA", oldpredict=res, user_image=imgPath)
    if result == 2:
        return render_template("result.html", data=stat, res="I think this game is a RTS", oldpredict=res, user_image=imgPath)

    return render_template("result.html", data=stat, res="Error during prediction", oldpredict=res, user_image=imgPath)


@app.route('/handle_data', methods=['POST'])
def handle_data():
    model = request.form['model']
    alpha = request.form['Alpha']
    alpha = float(alpha)
    epochs = request.form['Epochs']
    epochs = int(epochs)
    dataSetSize = request.form['Data set size']
    dataSetSize = int(int(dataSetSize) / 3)
    imageSize = request.form['Image size']
    imageSize = int(int(imageSize) / 2)
    prefix = request.form['Model Name']

    mlp_struct = request.form['mlp_struct']
    struct = [int(x.strip()) for x in mlp_struct.split(',')]

    
    # Lancer un train
    file_name = ""
    if model == "Linear Model":
        print("Linear Model Launching: \n\t" + " Nb images: " + str(dataSetSize * 3) + 
                                        "\n\t Image size: " + str(imageSize) + "x" + str(imageSize) +
                                        "\n\t Learning rate: " + str(alpha) + 
                                        "\n\t Epochs: " + str(epochs) + 
                                        "\n\t File name: " + prefix)
        file_name = fit_save_classif(dataSetSize, imageSize,
                         imageSize, alpha, epochs, prefix)
    elif model == "Linear Model Tensorflow":
        print("Linear Model Tensorflow Launching: \n\t" + " Nb images: " + str(dataSetSize * 3) + 
                                        "\n\t Image size: " + str(50) + "x" + str(50) +
                                        "\n\t File name: " + prefix)
        file_name = linear_keras(prefix, dataSetSize, 50, 50, epochs)
    elif model == "Multilayer perceptron":
        print("Multilayer perceptron Launching: \n\t" + " Nb images: " + str(dataSetSize * 3) + 
                                        "\n\t Image size: " + str(imageSize) + "x" + str(imageSize) +
                                        "\n\t Learning rate: " + str(alpha) + 
                                        "\n\t Epochs: " + str(epochs) + 
                                        "\n\t Layers: " + str(struct) +
                                        "\n\t File name: " + prefix)
        file_name = fit_save_mlp(dataSetSize, imageSize, imageSize,
                     alpha, epochs, prefix, struct)
    elif model == "RBF":
        print("RBF")
    res = getOldPredict()
    return render_template("trained.html", oldpredict=res, filename=file_name)


if __name__ == '__main__':
    buildModelFolders()
    app.run(debug=True, host='0.0.0.0', port=5000)

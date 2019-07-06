from flask import Flask, request, render_template, send_from_directory
import os
import sys
from flask_material import Material
from linear_dataset import fit_save_classif, load_predict_classif
from mlp_dataset import fit_save_mlp
from os import listdir
from os.path import isfile, join
import csv

app = Flask(__name__)
Material(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    myLinearPath = "Models/Linear/"
    myMlpPath = "Models/MLP/"
    myRBFPath = "Models/RBF/"
    res = []
    with open('static/prediction.csv') as csvfile:
        rows = csv.reader(csvfile)
        res.append(list(zip(*rows)))

    print(res[0])

    linearFiles = [f for f in listdir(
        myLinearPath) if isfile(join(myLinearPath, f))]
    mlpFiles = [f for f in listdir(myMlpPath) if isfile(join(myMlpPath, f))]
    RBFFiles = [f for f in listdir(myRBFPath) if isfile(join(myRBFPath, f))]
    return render_template('index.html', linear=linearFiles, mlp=mlpFiles, rbf=RBFFiles, oldpredict=res[0])


@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        dest = "/".join([target, "tempImg.PNG"])
        print(dest)
        file.save(dest)
    # Lancer une prédiction avec le bon model.
    # return le bon template (if [1,0,0] return FPS, [0,1,0] MOBA stc)
    #h, w, pathFPS, pathMOBA, pathRTS, imageToPredict

    Ypredict_FPS, Ypredict_MOBA, Ypredict_RTS, result = load_predict_classif(25, 25, "../PyTest/Models/Linear/linear_dataset_FPS_rendu3_15_30_500.model",
                                                                             "../PyTest/Models/Linear/linear_dataset_MOBA_rendu3_15_30_500.model",
                                                                             "../PyTest/Models/Linear/linear_dataset_RTS_rendu3_15_30_500.model",
                                                                             dest)

    stat = []
    stat.append(Ypredict_FPS)
    stat.append(Ypredict_MOBA)
    stat.append(Ypredict_RTS)
    if result == 0:
        return render_template("resultFPS.html", data=stat)
    if result == 1:
        return render_template("resultMOBA.html", data=stat)
    if result == 2:
        return render_template("resultRTS.html", data=stat)

    return render_template("resultFPS.html", data=stat)


@app.route('/handle_data', methods=['POST'])
def handle_data():
    #(1200, 100, 100, 0.05, 500, "100x100_1200_22h10")
    model = request.form['model']
    print(model)
    alpha = request.form['Alpha']
    alpha = float(alpha)
    epochs = request.form['Epochs']
    epochs = int(epochs)
    dataSetSize = request.form['Data set size']
    dataSetSize = int(int(dataSetSize) / 3)
    imageSize = request.form['Image size']
    imageSize = int(int(imageSize) / 2)
    prefix = request.form['Model Name']

    # Lancer un train
    if model == "Linear Model":
        fit_save_classif(dataSetSize, imageSize,
                         imageSize, alpha, epochs, prefix)
    elif model == "Multilayer perceptron":
        fit_save_mlp(dataSetSize, imageSize, imageSize,
                     alpha, epochs, prefix, [5, 5])
    elif model == "RBF":
        print("RBF")
    return render_template("trained.html")


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template, send_from_directory
import os

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template('index.html')

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
    return render_template("resultFPS.html")

@app.route('/handle_data', methods=['POST'])
def handle_data():
    alpha = request.form['Alpha']
    epochs = request.form['Epochs']
    dataSetSize = request.form['Data set size']
    imageSize = request.form['Image size']

    # Lancer un train
    return render_template("trained.html")
    

if __name__ == '__main__':
   app.run()
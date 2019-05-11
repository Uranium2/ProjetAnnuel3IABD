import os

from flask import Flask, request, render_template, send_from_directory

__author__ = 'Group6 3IABD'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

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
    
    return render_template("complete.html")

if __name__ == "__main__":
    app.run(port=4555, debug=True)
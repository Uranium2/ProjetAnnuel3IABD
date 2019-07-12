from PIL import Image
import os


def getDataSet(path, limit, h, w, isValidation):
    i = 0
    files = []
    exclude = []
    if not isValidation:
        exclude = ['FPS_Validation', 'MOBA_Validation', 'RTS_Validation']
    else:
        limit = 50
        exclude = ['FPS', 'MOBA', 'RTS']
    output = [[1,-1,-1],
                [-1,1,-1],
                [-1,-1,1]]
    YTrain = []
    yindex = 0
    # r=root, d=directories, f = files
    for r, directories, f in os.walk(path):
        directories[:] = [d for d in directories if d not in exclude]
        for file in f:
            if i >= limit:
                i = 0
                yindex = yindex + 1
                break
            if '.png' in file:
                files.append(os.path.join(r, file))
                i = i + 1
                for y in range(3):
                    YTrain.append(output[yindex][y])

    XTrain = []
    for f in files:
        im = Image.open(f)
        imResize = im.resize((h, w), Image.ANTIALIAS)
        imgLoad = imResize.load()
        for x in range(h):
            for y in range(w):
                R,G,B = imgLoad[x, y]
                XTrain.append(R / 255)
                XTrain.append(G / 255)
                XTrain.append(B / 255)
        im.close()

    return XTrain, YTrain

def getImgPath(path, limit, h, w, isValidation):
    i = 0
    files = []
    exclude = []
    if not isValidation:
        exclude = ['FPS_Validation', 'MOBA_Validation', 'RTS_Validation']
    else:
        limit = 50
        exclude = ['FPS', 'MOBA', 'RTS']



    # r=root, d=directories, f = files
    for r, directories, f in os.walk(path):
        directories[:] = [d for d in directories if d not in exclude]
        for file in f:
            if i >= limit:
                i = 0
                break
            if '.png' in file:
                files.append(os.path.join(r, file))
                i = i + 1

    return files

def save_stats(Model, Epochs, Alpha, Size, Data_set_size, struct, Accuracy_Set, Accurracy_validation):
    myCsvRow = str(Model) + "," + str(Epochs) + "," + str(Alpha) + "," + str(Size) + "," + str(Data_set_size) + "," + ';'.join(map(str, struct))  + "," +  "{:.2f}".format(Accuracy_Set) + "%," + "{:.2f}".format(Accurracy_validation) + "%\n"

    with open('static/prediction.csv','a') as fd:
        fd.write(myCsvRow)
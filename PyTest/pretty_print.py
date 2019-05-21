from dll_load import predict_regression
import matplotlib.pyplot as plt



def predict_2D(W, inputCountPerSample):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for x in range(0, 200):
        for y in range(0, 200):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res = predict_regression(W, dot, inputCountPerSample)
            if ( res > 0):
                x1.append(x / 100)
                y1.append(y / 100)
            else:
                x2.append(x / 100)
                y2.append(y / 100)

    plt.scatter(x1, y1, c = 'green')
    plt.scatter(x2, y2, c = 'red')
    plt.show()


def predict_2D_OR(W, inputCountPerSample):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for x in range(0, 100):
        for y in range(0, 100):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res = predict_regression(W, dot, inputCountPerSample)
            if ( res > 0):
                x1.append(x / 100)
                y1.append(y / 100)
            else:
                x2.append(x / 100)
                y2.append(y / 100)


    x3 = [0]
    y3 = [0]
    x4 = [0, 1, 1]
    y4 = [1, 0, 1]
    plt.scatter(x1, y1, c = 'green')
    plt.scatter(x2, y2, c = 'red')
    plt.scatter(x3, y3, c = 'yellow')
    plt.scatter(x4, y4, c = 'magenta')
    plt.show()

def predict_2D_XOR(W, inputCountPerSample):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for x in range(0, 100):
        for y in range(0, 100):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res = predict_regression(W, dot, inputCountPerSample)
            if ( res > 0):
                x1.append(x / 100)
                y1.append(y / 100)
            else:
                x2.append(x / 100)
                y2.append(y / 100)


    x3 = [0, 1]
    y3 = [0, 1]
    x4 = [0, 1]
    y4 = [1, 0]
    plt.scatter(x1, y1, c = 'green')
    plt.scatter(x2, y2, c = 'red')
    plt.scatter(x3, y3, c = 'yellow')
    plt.scatter(x4, y4, c = 'magenta')
    plt.show()

def predict_2D_AND(W, inputCountPerSample):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for x in range(0, 100):
        for y in range(0, 100):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res = predict_regression(W, dot, inputCountPerSample)
            if ( res > 0):
                x1.append(x / 100)
                y1.append(y / 100)
            else:
                x2.append(x / 100)
                y2.append(y / 100)


    x3 = [0, 0, 1]
    y3 = [0, 1, 0]
    x4 = [1]
    y4 = [1]
    plt.scatter(x1, y1, c = 'green')
    plt.scatter(x2, y2, c = 'red')
    plt.scatter(x3, y3, c = 'yellow')
    plt.scatter(x4, y4, c = 'magenta')
    plt.show()


def predict_2D_3Class(W1, W2, W3, inputCountPerSample,x3, y3, x4, y4, x5, y5):
    x11 = []
    y11 = []
    x12 = []
    y12 = []
    x13 = []
    y13 = []

    for x in range(0, 200):
        for y in range(0, 200):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res1 = predict_regression(W1, dot, inputCountPerSample)
            res2 = predict_regression(W2, dot, inputCountPerSample)
            res3 = predict_regression(W3, dot, inputCountPerSample)
            l = []
            l.append(res1)
            l.append(res2)
            l.append(res3)
            if ( res1 == max(l)):
                x11.append(x / 100)
                y11.append(y / 100)
            if ( res2 == max(l)):
                x12.append(x / 100)
                y12.append(y / 100)
            if ( res3 == max(l)):
                x13.append(x / 100)
                y13.append(y / 100)



    plt.scatter(x11, y11, c = 'green')
    plt.scatter(x12, y12, c = 'red')
    plt.scatter(x13, y13, c = 'cyan')
    
    plt.scatter(x3, y3, c = 'yellow')
    plt.scatter(x4, y4, c = 'magenta')
    plt.scatter(x5, y5, c = 'blue')
    plt.show()

def predict_2D_3Class_individual(W, inputCountPerSample,x3, y3, x4, y4, x5, y5):
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for x in range(0, 200):
        for y in range(0, 200):
            dot = []
            dot.append(x / 100)
            dot.append(y / 100)
            res = predict_regression(W, dot, inputCountPerSample)
            if ( res > 0):
                x1.append(x / 100)
                y1.append(y / 100)
            else:
                x2.append(x / 100)
                y2.append(y / 100)



    plt.scatter(x1, y1, c = 'green')
    plt.scatter(x2, y2, c = 'red')
    
    plt.scatter(x3, y3, c = 'yellow')
    plt.scatter(x4, y4, c = 'magenta')
    plt.scatter(x5, y5, c = 'blue')
    plt.show()
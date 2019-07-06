from dll_load import get_Kmeans, flatten
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# generate 2d classification dataset
Xx, Y = make_moons(n_samples=100, noise=0)
Xy = list(flatten(Xx))
X = []
Y = []
Xk = []
Yk = []

K = 3
XTrain = [
    0,
    0,
    0.3,
    1,
    0.5,
    0.5,
    0,
    0.5,
    1,
    1,
    4,
    4,
    4,
    5,
    4.5,
    6,
    0,
    6,
    0.5,
    7,
    0.2,
    6.5,
    0.5,
    7.1,
    0,
    5,
]
inputCountPerSample = 2
sampleCount = int(len(XTrain) / inputCountPerSample)
epochs = 400

Kmeans = get_Kmeans(K, XTrain, sampleCount, inputCountPerSample, epochs)


for x, y in zip(XTrain[0::2], XTrain[1::2]):
    X.append(x)
    Y.append(y)

for x, y in zip(Kmeans[0::2], Kmeans[1::2]):
    Xk.append(x)
    Yk.append(y)

plt.scatter(X, Y, c="red")
plt.scatter(Xk, Yk, c="green")
plt.show()


print(Kmeans)

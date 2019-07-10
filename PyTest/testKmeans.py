from dll_load import get_Kmeans, flatten, fit_regRBF_Kmeans
from pretty_print import predict_2D_RBF_K
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy

# generate 2d classification dataset
X = []
Y = []
Xk = []
Yk = []
XTrain = []

K = 4
gamma = 1

# generate 2d classification dataset
Xx, YTrain = make_blobs(n_samples=200, centers=K, cluster_std=0.4, random_state=1)


XTrain = list(flatten(Xx))


inputCountPerSample = 2
sampleCount = int(len(XTrain) / inputCountPerSample)
epochs = 1000

Kmeans = get_Kmeans(K, XTrain, sampleCount, inputCountPerSample, epochs)


for x, y in zip(XTrain[0::2], XTrain[1::2]):
    X.append(x)
    Y.append(y)

for x, y in zip(Kmeans[0::2], Kmeans[1::2]):
    Xk.append(x)
    Yk.append(y)

plt.scatter(X, Y, c = 'red')
plt.scatter(Xk, Yk, c = 'green')
plt.show()
W = fit_regRBF_Kmeans(Kmeans, K, XTrain, YTrain, sampleCount, inputCountPerSample, gamma)
print(W)
predict_2D_RBF_K(W, XTrain, YTrain, inputCountPerSample, gamma, sampleCount, 0, 10, K)

import numpy as np
import tensorflow.keras as tf


if __name__ == "__main__":
    print("multi layer perceptron")

    # DATASET
    x = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    y = np.concatenate(
        [np.ones((50, 2)), np.ones((50, 2)) * -1.0])

    # MODEL
    model = tf.Sequential()

    # LAYERS


    # COMPILE


    # FIT

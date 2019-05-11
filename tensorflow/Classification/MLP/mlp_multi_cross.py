import numpy as np
import tensorflow.keras as tf


if __name__ == "__main__":
    print("multi layer perceptron")

    x = np.random.random((1000, 2)) * 2.0 - 1.0
    y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(
        p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in x])

    # MODEL
    model = tf.Sequential()

    # LAYERS


    # COMPILE


    # FIT

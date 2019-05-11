import numpy as np
import tensorflow.keras as tf

if __name__ == "__main__":
    print("linear cross model")

    # DATASET
    x = np.random.random((500, 2)) * 2.0 - 1.0
    y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in x])

    # MODEL
    model = tf.Sequential()

    # LAYERS


    # COMPILE


    # FIT

import numpy as np
import tensorflow.keras as tf


if __name__ == "__main__":
    print("multi layer perceptron")

    x = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ])
    y = np.array([
        1,
        -1,
        -1
    ])



    model = tf.Sequential()

    model.add(tf.layers.Dense(1, input_dim=20, activation='relu'))
    model.add(tf.layers.Dense(1, activation='relu'))
    model.add(tf.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='SGD',
                  metrics=['accuracy'])

    if model.fit(x, y, epochs=100):
        model.fit(x, y, epochs=100)
        print("\nmlp works successfully")
    else:
        print("error")
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

    model.add(tf.layers.Dense(64, input_dim=20, activation='relu'))
    model.add(tf.layers.Dense(64, activation='relu'))
    model.add(tf.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.fit(x, y, epochs=100)


import numpy as np
import tensorflow.keras as tf


if __name__ == "__main__":
    print("multi layer perceptron")

    x = np.array([
        [-1, -1],
        [1, 1],
        [1, 0],
    ])
    y = np.array([
        0,
        0,
        1,
        1
    ])

    x_train = np.random.random((1000, 20))
    y_train = np.random.randint(2, size=(1000, 1))

    model = tf.Sequential()

    model.add(tf.layers.Dense(1, input_dim=20, activation='relu'))
    model.add(tf.layers.Dense(1, activation='relu'))
    model.add(tf.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='SGD',
                  metrics=['accuracy'])

    if model.fit(x_train, y_train, epochs=100):
        model.fit(x_train, y_train, epochs=100)
        print("mlp works successfully")
    else:
        print("error")
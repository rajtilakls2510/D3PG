# from tensorflow.keras import Model
# import tensorflow.keras.layers as layers
#
# inp = layers.Input(shape=(512, ))
# x = layers.Dense(64, activation = "relu")(inp)
# x = layers.Dense(64, activation = "relu")(x)
# x = layers.Dense(64, activation = "relu")(x)
# out = layers.Dense(10, activation = "softmax")(x)
#
# model = Model(inputs = inp, outputs = out)
#
# model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#
# print(model.summary())

import socket
print(type(socket.gethostbyname(socket.gethostname())))
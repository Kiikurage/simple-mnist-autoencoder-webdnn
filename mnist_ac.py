# -*- coding: utf-8 -*-
#

import argparse
import json, os

from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D, Input, add, GlobalAveragePooling2D, Activation
from keras.models import Sequential, Model

def create_model():
    model = Sequential()
    input_shape = (28, 28, 1)
    model.add(Conv2D(8, kernel_size=3, activation="relu", input_shape=input_shape))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2DTranspose(1, kernel_size=3))
    model.add(Activation("sigmoid"))
    return model
    

def main():
    from keras.datasets import mnist
    # shape: (size, 28, 28), uint8
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32")
    x_train /= 255
    s, w, h = x_train.shape
    x_train = x_train.reshape(s, w, h, 1)
    model = create_model()
    #import pdb; pdb.set_trace()
    model.summary()
    model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
    model.fit(x_train, x_train, batch_size=10, epochs=2, verbose=1)
    model.save("autoencoder.h5")
    model.save_weights("weights_ac.h5")

if __name__ == '__main__':
    main()

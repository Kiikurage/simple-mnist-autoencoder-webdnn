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
    model.load_weights("weights_ac.h5")
    #import pdb; pdb.set_trace()
    model.summary()
    inp = x_train[0]
    inp = inp.reshape(1, 28, 28, 1)
    ret = model.predict(inp, batch_size=1)
    #import pdb; pdb.set_trace()
    #print(ret)
    import matplotlib.pyplot as plt
    #fig = plt.figure()
    plt.subplot(1, 2, 1)
    img1 = inp.reshape(28, 28)
    plt.imshow(img1)
    
    ret = ret.reshape(28, 28)
    plt.subplot(1, 2, 2)
    plt.imshow(ret)
    plt.show()

def load_data():
    pass

if __name__ == '__main__':
    main()

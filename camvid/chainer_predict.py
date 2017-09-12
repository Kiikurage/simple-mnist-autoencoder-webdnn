# -*- coding: utf-8 -*-
#

import argparse
import chainer
import numpy as np

import chainer.functions as F
import chainer.links as L
from chainer import reporter
from chainer import training
from chainer.training import extensions

from chainercv.datasets import CamVidDataset
from chainercv.utils import read_image

class CamVidAutoEncoder(chainer.Chain):
    def __init__(self, loss_func):
        self.channel = 3
        self.width = 360
        self.height = 480
        self.loss_func = loss_func
        super(CamVidAutoEncoder, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(self.channel, 16, ksize=3)
            self.conv2 = L.Convolution2D(16, 32, ksize=3)
            self.deconv1 = L.Deconvolution2D(32, 16, ksize=3)
            self.deconv2 = L.Deconvolution2D(16, self.channel, ksize=3)

    def predict(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = self.deconv1(h)
        h = F.relu(h)
        h = self.deconv2(h)
        h = F.tanh(h)
        return h

    def __call__(self, x, t):
        y = self.predict(x)
        loss = self.loss_func(y, t)
        reporter.report({'loss': loss})
        return loss

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.npz')
    parser.add_argument('--out', type=str, default='out')
    args = parser.parse_args()
    return args

def main():
    args = arg()
    model = CamVidAutoEncoder(F.mean_squared_error)
    chainer.serializers.load_npz(args.model, model)

    train = CamVidDataset(split='test')
    img = train[0][0]
    img /= 255.0
    ch, w, h = img.shape
    example_input = img.reshape(1, ch, w, h)
    x = chainer.Variable(example_input)
    y = model.predict(x)

    import matplotlib.pyplot as plt
    img1 = img.transpose(1, 2, 0)
    plt.subplot(1,2,1)
    plt.imshow(img1)
    y = y.data.reshape(ch, w, h)
    y = y.transpose(1, 2, 0)
    plt.subplot(1,2,2)
    plt.imshow(y)
    plt.show()

if __name__ == '__main__':
    main()

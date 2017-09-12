# -*- coding: utf-8 -*-
import chainer
import numpy as np

import chainer.functions as F
import chainer.links as L
from chainer import reporter

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
            self.conv3 = L.Convolution2D(32, 64, ksize=3)
            self.deconv1 = L.Deconvolution2D(64, 32, ksize=3)
            self.deconv2 = L.Deconvolution2D(32, 16, ksize=3)
            self.deconv3 = L.Deconvolution2D(16, self.channel, ksize=3)

    def predict(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = self.conv3(h)
        h = F.relu(h)
        h = self.deconv1(h)
        h = F.relu(h)
        h = self.deconv2(h)
        h = F.relu(h)
        h = self.deconv3(h)
        h = F.tanh(h)
        return h

    def __call__(self, x, t):
        y = self.predict(x)
        loss = self.loss_func(y, t)
        reporter.report({'loss': loss})
        return loss

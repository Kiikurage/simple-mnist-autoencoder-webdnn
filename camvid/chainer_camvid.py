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

class MyCamVidDataset(CamVidDataset):
    def __init__(self, **kwargs):
        super(MyCamVidDataset, self).__init__(**kwargs)

    def get_example(self, i):
        if i >= len(self):
            raise IndexError('index too large')
        img_filename, _ = self.filenames[i]
        img = read_image(img_filename, color=True)
        img -= 127.5
        img /= 127.5
        return img, img

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
            self.deconv1 = L.Deconvolution2D(16, self.channel, ksize=3)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=12)
    parser.add_argument('--out', type=str, default='out')

    train = MyCamVidDataset(split='train') # shape (3, 360, 480)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
    

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
from net import CamVidAutoEncoder

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
    y = y.transpose(1, 2, 0).clip(0, 1.0)
    plt.subplot(1,2,2)
    plt.imshow(y)
    plt.show()

if __name__ == '__main__':
    main()

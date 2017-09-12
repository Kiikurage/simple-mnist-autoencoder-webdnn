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

from net import CamVidAutoEncoder
from webdnn.backend import generate_descriptor
from webdnn.frontend.chainer import ChainerConverter

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

    example_input = np.zeros((1, 3, 480, 360), dtype=np.float32)
    x = chainer.Variable(example_input)
    y = model.predict(x)
    graph = ChainerConverter().convert([x], [y])
    for backend in ["webgpu", "webassembly"]:
        exec_info = generate_descriptor(backend, graph)
        exec_info.save(args.out)

if __name__ == '__main__':
    main()

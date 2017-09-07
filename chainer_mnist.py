# -*- coding: utf-8 -*-

import os
import argparse
import json

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

class AutoEncoder(chainer.Chain):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 8, ksize=3)
            self.deconv1 = L.Deconvolution2D(8, 1, ksize=3)

    def __call__(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        h = self.deconv1(h)
        y = F.sigmoid(h)
        return y

class MyMnist(chainer.dataset.DatasetMixin):
    def __init__(self):
        train, test = chainer.datasets.get_mnist(ndim=3)
        self.train = train
        self.test = test
    def __len__(self):
        return len(self.train)
    
    def __getitem__(self, i):
        #import pdb; pdb.set_trace()
        return self.train[i][0]

class MyAutoEncoderUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop('model')
        self.device = kwargs.pop('device')
        self.img_size = 28
        super(MyAutoEncoderUpdater, self).__init__(*args, device=self.device, *kwargs)

    def update_core(self):
        model = self.model
        xp = model.xp
        opt = self.get_optimizer('main')
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x_in = xp.zeros((batchsize, 1, self.img_size, self.img_size)).astype("f")
        t_out = xp.zeros((batchsize, 1, self.img_size, self.img_size)).astype("f")
        for i in range(batchsize):
            x_in[i,:] = xp.asarray(batch[i])
            t_out[i,:] = xp.asarray(batch[i])
        x_in = chainer.Variable(x_in)
        x_out = model(x_in)
        import pdb; pdb.set_trace()
        r = opt.update(self.loss_func, model, x_out, t_out)
        x_in.unchain_backward()
        x_out.unchain_backward()
        return
    
def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=2)
    args = parser.parse_args()
    return args

def main():
    args = arg()
    my_mnist = MyMnist()
    train_iter = chainer.iterators.SerialIterator(my_mnist, args.batchsize)
    #import pdb; pdb.set_trace()
    model = AutoEncoder()
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)
    updater = MyAutoEncoderUpdater(train_iter, optimizer, loss_func=F.mean_squared_error, device=args.gpu, model=model)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
    trainer.run()

if __name__ == '__main__':
    main()

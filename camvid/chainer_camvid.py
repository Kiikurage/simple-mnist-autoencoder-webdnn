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
from chainer import cuda

from net import CamVidAutoEncoder

class MyCamVidDataset(CamVidDataset):
    def __init__(self, **kwargs):
        self.pos = -1
        super(MyCamVidDataset, self).__init__(**kwargs)

    def get_example(self, i):
        if i >= len(self):
            raise IndexError('index too large')
        img_filename, _ = self.filenames[i]
        img = read_image(img_filename, color=True)
        img /= 255.0
        return img

    def next(self):
        self.pos += 1
        return self.get_example(self.pos)

    def reset(self):
        self.pos = -1

class CamViidAEIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos = 0
        self.epoch = 0

    def fetch(self):
        data = []
        try:
            for i in range(self.batch_size):
                data.append(self.dataset.next())
            self.pos += self.batch_size
        except:
            self.pos = 0
            self.epoch += 1
            self.dataset.reset()
        return data

    def __next__(self):
        data = self.fetch()
        if len(data) <= 0:
            data = self.fetch()
        data = np.asarray(data)
        return data, data.copy()

    @property
    def epoch_detail(self):
        ed = self.epoch + float(self.pos / len(self.dataset))
        return ed

def convert(batch, device):
    if device >= 0:
        x = cuda.to_gpu(batch[0])
        t = cuda.to_gpu(batch[1])
        return x, t
    return batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--batchsize', type=int, default=12)
    parser.add_argument('--out', type=str, default='out')
    parser.add_argument('--epoch', type=int, default=5)
    args = parser.parse_args()

    # trigger
    log_trigger = (50, 'iteration')

    train = MyCamVidDataset(split='train') # shape (3, 360, 480)
    train_iter = CamViidAEIterator(train, args.batchsize)
    model = CamVidAutoEncoder(F.mean_squared_error)
    opt = chainer.optimizers.MomentumSGD()
    opt.setup(model)
    updater = training.StandardUpdater(train_iter, opt, converter=convert,
                                       device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=log_trigger))
    trainer.extend(extensions.PrintReport(['epoch', 'loss']), trigger=log_trigger)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.run()
    chainer.serializers.save_npz("model.npz", model)


if __name__ == '__main__':
    main()
    

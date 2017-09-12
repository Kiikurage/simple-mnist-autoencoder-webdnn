# -*- coding: utf-8 -*-
#

import argparse
import chainer
import numpy as np

from chainercv.datasets import CamVidDataset

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.npz')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--index', '-i', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = arg()
    data = CamVidDataset(split=args.type)
    img = data[args.index][0]
    import pdb; pdb.set_trace()
    img /= 255.0
    #img = np.clip(img, -1.0, 1.0)
    import matplotlib.pyplot as plt
    img1 = img.transpose(1, 2, 0)
    plt.imshow(img1)
    plt.show()

if __name__ == '__main__':
    main()

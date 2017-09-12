# -*- coding: utf-8 -*-

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    width = 480
    height = 360
    ch = 3

    import json
    with open(args.input) as f:
        jv = json.load(f)
    import numpy as np
    img = np.zeros((width * height * ch), dtype=np.float32)
    for i in range(len(jv)):
        si = str(i)
        v = jv[si]
        img[i] = v
    img = img.reshape(ch, width, height).transpose(2, 1, 0)
    import pdb; pdb.set_trace()
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

main()

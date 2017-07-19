"""
Author: Shaobo Cui:
refer:
1. https://github.com/pytorch/vision/blob/master/torchvision/utils.py
2. https://github.com/carpedm20/BEGAN-tensorflow/blob/master/utils.py
"""

from __future__ import division

import math
from glob import glob

import numpy as np
from PIL import Image

def make_grid(in_arr, x_num, padding_up=2, padding_down=2, padding_left=2, padding_right=2):
    """
    It these function is to arange the ndarr (with shape [N, H, W, C])into the
    the an ndarray that can be used to write into picture. to better present it
    Add some padding to the around of each small pictures
    Eg. N = 64, and n_row = 8, the output will be 8 * 8 small pictures,
    each small picture will be added to small padding / 2
    :param in_arr: the input array with size of [N, H, W, C]
    :param x_num: the number of small pic located in each row
    :param padding: the padding around each small images.
    :return: return the grid
    """
    k = 0
    n_pics = in_arr.shape[0]
    height, width = in_arr.shape[1], in_arr.shape[2]
    channel = in_arr.shape[3]

    x_pics = min(x_num, n_pics)
    y_pics = int(math.ceil(n_pics / x_pics))

    height_pad = height + padding_up + padding_down
    width_pad = width + padding_left + padding_right

    grid = np.zeros([height_pad * y_pics, width_pad * x_pics, channel], dtype=np.uint8)

    for y in xrange(y_pics):
        for x in xrange(x_pics):
            if k > n_pics:
                break
            height_start = y * height_pad + padding_up
            width_start = x * width_pad + padding_left
            grid[height_start : height_start + height, width_start : width_start + width] = in_arr[k]
            k += 1
    return grid

def load_raw_image(filepath, ext, batch_size):
    filelist = glob("{}/*.{}".format(filepath, ext))[:batch_size]
    raw_batch = np.array([np.array(Image.open(fname)) for fname in filelist])
    return raw_batch




def save_image(raw, filename, x_num):
    grid = make_grid(raw, x_num=x_num)
    image = Image.fromarray(grid)
    image.save(filename)

filepath = './first_celeba';
ext = 'jpg'
batch_size = 64

raw_batch = load_raw_image(filepath, ext, 64)
save_image(raw_batch, 'multi.jpg', 8)






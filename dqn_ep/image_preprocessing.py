__author__ = 'frankhe'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def resize(image, size):
    return scipy.misc.imresize(image, size=size)


def imshow(photo, gray=False):
    if gray:
        plt.imshow(photo, cmap = plt.get_cmap('gray'))
    else:
        plt.imshow(photo)
    plt.show()


if __name__ == '__main__':
    img = mpimg.imread('wallpaper.jpg')
    gray = rgb2gray(img)
    gray = resize(gray, (1000, 1000))
    imshow(gray, True)

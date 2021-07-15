"""
Adapted from original code by Frank et al., 2020:
https://github.com/RUB-SysSec/GANDCTAnalysis
"""

import numpy as np
from PIL import Image
from scipy import fftpack


def load_image(path, greyscale=True, tf=False):
    """
    Loads an image in numpy array format.
    Arguments:
        path : str
            Image filepath.
        greyscale : bool
            Converts image to greyscale. The default is True.
    """
    x = Image.open(path)
    if greyscale:
        x = x.convert("L")
        if tf:
            x = np.asarray(x)
            x = np.reshape(x, [*x.shape, 1])
    return np.asarray(x)


def scale_image(image):
    """
    Normalizes an RGB image (8 bits/channel) to intensities [0, 1]
    """
    if not image.flags.writeable:
        image = np.copy(image)
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    image /= 127.5
    image -= 1.0
    return image


def dct2(array):
    """
    Uses the scipy fftpack implementation of the 2D discrete cosine transform
    to obtain the frequency domain representation of a greyscale image.
    """
    array = fftpack.dct(array, type=2, norm="ortho", axis=0)
    array = fftpack.dct(array, type=2, norm="ortho", axis=1)
    return array


def normalize(image, mean, std):
    """
    Normalizes an image/tensor given its mean values and standard deviations.
    """
    image = (image - mean) / std
    return image

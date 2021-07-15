"""
Adapted from original code by Frank et al., 2020:
https://github.com/RUB-SysSec/GANDCTAnalysis
"""

import tensorflow as tf


def dct2_tf(array, batched=True):
    
    """
    Transforms an array into its frequency domain representation using the
    2D discrete cosine transform (DCT) implemented in TensorFlow, 
    by first applying DCT along rows followed by columns.
    """
    
    shape = array.shape
    dtype = array.dtype
    array = tf.cast(array, tf.float32)

    if batched:
        # tensorflow computes over last axis (-1)
        # layout (B)atch, (R)ows, (C)olumns, (V)alue
        # BRCV
        array = tf.transpose(array, perm=[0, 3, 2, 1])
        # BVCR
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[0, 1, 3, 2])
        # BVRC
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[0, 2, 3, 1])
        # BRCV
    else:
        # RCV
        array = tf.transpose(array, perm=[2, 1, 0])
        # VCR
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[0, 2, 1])
        # VRC
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[1, 2, 0])
        # RCV

    array = tf.cast(array, dtype)

    array.shape.assert_is_compatible_with(shape)

    return array
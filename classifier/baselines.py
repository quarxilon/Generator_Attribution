"""
Adapted from original code by Frank et al., 2020:
https://github.com/RUB-SysSec/GANDCTAnalysis
"""

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, initializers


def layer_name(prefix, name, number):
    """
    Helper function to set standard layer names.
    """
    return f"{prefix}{name}_{number:02d}"


def build_gandctanalysis_simple_cnn(input_shape, num_classes, init, prefix="bl"):
    
    """
    Assembles the simple convolutional attribution network used by Frank et al. (2020).

    Parameters
    ----------
    input_shape : (*int)
        Input image tensor shape.
    num_classes : int
        Number of classes specified for multi-class attribution.
    init : keras.initializers.Initializer
        Weight initializer.
    prefix : str, optional
        Prefix added to all layer names. The default is "bl" (baseline).

    Returns
    -------
    model : keras.Model
        Constructed classifier model.

    """
    
    inputs = keras.Input(shape=input_shape, name=f"{prefix}Input")
    
    # 128x128
    x = layers.Conv2D(3, 3, padding="same", activation="relu", 
                      kernel_initializer=init, name=layer_name(prefix,"Conv",1))(inputs)
    x = layers.Conv2D(8, 3, padding="same", activation="relu",
                      kernel_initializer=init, name=layer_name(prefix,"Conv",2))(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), name=layer_name(prefix,"APool",2))(x)
    
    # 64x64
    x = layers.Conv2D(16, 3, padding="same", activation="relu",
                      kernel_initializer=init, name=layer_name(prefix,"Conv",3))(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), name=layer_name(prefix,"APool",3))(x)
    
    # 32x32
    x = layers.Conv2D(32, 3, padding="same", activation="relu",
                      kernel_initializer=init, name=layer_name(prefix,"Conv",4))(x)

    x = layers.Flatten()(x)
    if num_classes == 1: # deepfake detection or binary attribution setting
        activation = "sigmoid"
    else: # multi-class attribution setting
        activation = "softmax"
    outputs = layers.Dense(num_classes, activation=activation, 
                           kernel_initializer=init, name=f"{prefix}Decision")(x)
    
    model = keras.Model(inputs, [outputs])
    return model


def build_ganfingerprints_postpool_cnn(input_shape, num_classes, init, alpha=0.2, prefix="bl"):
    
    """
    Assembles a postpooling variant of the attribution network used by
    Yu, Davis, & Fritz (2019): https://github.com/ningyu1991/GANFingerprints
    Postpooling starts at 16x16 image resolution.
    The number of neurons/convolutional kernels per layer is one quarter of the original size.

    Parameters
    ----------
    input_shape : (*int)
        Input image tensor shape.
    num_classes : int
        Number of classes specified for multi-class attribution.
    init : keras.initializers.Initializer
        Weight initializer.
    alpha : float
        LeakyReLU "leakage" multiplier value.
    prefix : str, optional
        Prefix added to all layer names. The default is "bl" (baseline).

    Returns
    -------
    model : keras.Model
        Constructed classifier model.

    """
    
    inputs = keras.Input(shape=input_shape, name=f"{prefix}Input")
    res = int(np.log2(input_shape[1])) # for 128, this equals 7
    
    # 128x128
    x = layers.Conv2D(4, 3, padding="same", kernel_initializer=init, 
                      name=layer_name(prefix,"Conv",1))(inputs)
    x = layers.LeakyReLU(alpha=alpha, name=layer_name(prefix,"LRelu",1))(x)
    x = layers.Conv2D(8, 3, padding="same", kernel_initializer=init, 
                      name=layer_name(prefix,"Conv",2))(x)
    x = layers.LeakyReLU(alpha=alpha, name=layer_name(prefix,"LRelu",2))(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), name=layer_name(prefix,"APool",2))(x)
    res -= 1
    
    # 64x64
    x = layers.Conv2D(8, 3, padding="same", kernel_initializer=init, 
                      name=layer_name(prefix,"Conv",3))(x)
    x = layers.LeakyReLU(alpha=alpha, name=layer_name(prefix,"LRelu",3))(x)
    x = layers.Conv2D(16, 3, padding="same", kernel_initializer=init, 
                      name=layer_name(prefix,"Conv",4))(x)
    x = layers.LeakyReLU(alpha=alpha, name=layer_name(prefix,"LRelu",4))(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), name=layer_name(prefix,"APool",4))(x)
    res -= 1
    
    # 32x32
    x = layers.Conv2D(16, 3, padding="same", kernel_initializer=init, 
                      name=layer_name(prefix,"Conv",5))(x)
    x = layers.LeakyReLU(alpha=alpha, name=layer_name(prefix,"LRelu",5))(x)
    x = layers.Conv2D(32, 3, padding="same", kernel_initializer=init, 
                      name=layer_name(prefix,"Conv",6))(x)
    x = layers.LeakyReLU(alpha=alpha, name=layer_name(prefix,"LRelu",6))(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), name=layer_name(prefix,"APool",6))(x)
    res -= 1
    
    # 32x32
    block_count = 7
    while res > 2:
        x = layers.AveragePooling2D(pool_size=(2, 2), name=layer_name(prefix,"APool",block_count))(x)
        block_count += 1
        res -= 1
        
    # 4x4
    x = layers.Conv2D(128, 3, padding="same", kernel_initializer=init, 
                      name=layer_name(prefix,"Conv",block_count))(x)
    x = layers.LeakyReLU(alpha=alpha, name=layer_name(prefix,"LRelu",block_count))(x)
    block_count += 1
    
#    x = layers.Conv2D(256, 1, padding="valid", kernel_initializer=init, 
#                      name=layer_name(prefix,"Conv",block_count))(x)
#    x = layers.LeakyReLU(alpha=alpha, name=layer_name(prefix,"LRelu",block_count))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, kernel_initializer=init, name=layer_name(prefix,"Dense",block_count))(x)
    x = layers.LeakyReLU(alpha=alpha, name=layer_name(prefix,"LRelu",block_count))(x)
    
    if num_classes == 1: # deepfake detection or binary attribution setting
        activation = "sigmoid"
    else: # multi-class attribution setting
        activation = "softmax"
        
    outputs = layers.Dense(num_classes, activation=activation,
                           kernel_initializer=init, name=f"{prefix}Decision")(x)
    
#    x = layers.Conv2D(num_classes, 1, padding="valid", activation=activation, 
#                      kernel_initializer=init, name=f"{prefix}PatchDecision")(x)
#    if num_classes == 1:
#        x = layers.MaxPooling2D(pool_size=(4,4), name=f"{prefix}PatchMaxPool")(x)
#    else:
#        x = layers.AveragePooling2D(pool_size=(4,4), name=f"{prefix}PatchAPool")(x)
#    outputs = layers.Flatten()(x)
    
    model = keras.Model(inputs, [outputs])
    return model


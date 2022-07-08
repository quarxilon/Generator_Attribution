import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, initializers

"""
Expected input_shape:
    Pixel: [batch_size, 256, 256, 3] or [batch_size, 128, 128, 3]
    DCT: [batch_size, 256, 256, 1] or [batch_size, 128, 128, 1]
    
n_out = [(n_in + 2{padding} - {kernel}) / {strides}] + 1

2 rounds of convolutional layers before downsampling:
    1. layers.Conv2D(fmaps, 3, strides=1, padding="same") same resolution
    2. layers.Conv2D(fmaps, 4, strides=2, padding="same") downsample 2x

3 rounds of downsampling before post-pooling is enough
Bigger patch coverage known to give diminishing returns
CHANGELOG 2021/03/05: try increasing to 4 rounds of downsampling with 64 feature maps
CHANGELOG 2021/03/12: revert to 3 rounds of downsampling
"""


def layer_name(prefix, name, number):
    """
    Helper function to set standard layer names.
    """
    return f"{prefix}{name}_{number:02d}"


def build_primary_layers(input_shape, init, prefix="d"):
    
    """
    Assembles primary convolutional module, used for deepfake detection.
    DOES NOT PRODUCE A FULL NEURAL NET!
    
    Parameters
    ----------
    input_shape : (*int)
        Input image tensor shape.
    init : keras.initializers.Initializer
        Weight initializer.
    prefix : str, optional
        Prefix added to all layer names. The default is "d".

    Returns
    -------
    inputs : keras.Input
        Input layer (for model construction via Keras functional API)
    det_features : keras.layers.Layer
        Complete primary feature extraction outputs.
    mid_features : keras.layers.Layer
        Intermediate feature extraction outputs, for secondary layers.
        
    """
    
    inputs = keras.Input(shape=input_shape, name=f"{prefix}Input")
    
    # 256x256
    x = layers.Conv2D(3, 3, strides=1, padding="same", 
                      kernel_initializer=init, 
                      name=layer_name(prefix,"Conv",1))(inputs)
    x = layers.ReLU(name=layer_name(prefix,"Relu",1))(x)
    x = layers.Conv2D(8, 4, strides=2, padding="same", 
                      kernel_initializer=init,
                      name=layer_name(prefix,"Conv",2))(x)
    x = layers.BatchNormalization(name=layer_name(prefix,"BNorm",2))(x)
    x = layers.ReLU(name=layer_name(prefix,"Relu",2))(x)
    
    # 128x128
    x = layers.Conv2D(16, 3, strides=1, padding="same", 
                      kernel_initializer=init,
                      name=layer_name(prefix,"Conv",3))(x)
    x = layers.BatchNormalization(name=layer_name(prefix,"BNorm",3))(x)
    x = layers.ReLU(name=layer_name(prefix,"Relu",3))(x)
    x = layers.Conv2D(16, 4, strides=2, padding="same", 
                      kernel_initializer=init,
                      name=layer_name(prefix,"Conv",4))(x)
    x = layers.BatchNormalization(name=layer_name(prefix,"BNorm",4))(x)
    mid_features = layers.ReLU(name=layer_name(prefix,"Relu",4))(x)
    
    # 64x64
    x = layers.Conv2D(32, 3, strides=1, padding="same", 
                      kernel_initializer=init,
                      name=layer_name(prefix,"Conv",5))(mid_features)
    x = layers.BatchNormalization(name=layer_name(prefix,"BNorm",5))(x)
    x = layers.ReLU(name=layer_name(prefix,"Relu",5))(x)
    x = layers.Conv2D(32, 4, strides=2, padding="same", 
                      kernel_initializer=init,
                      name=layer_name(prefix,"Conv",6))(x)
    x = layers.BatchNormalization(name=layer_name(prefix,"BNorm",6))(x)
    # x = layers.ReLU(name=layer_name(prefix,"Relu",6))(x)
    det_features = layers.ReLU(name=layer_name(prefix,"Relu",6))(x)
    
    """
    # 32x32 
    x = layers.Conv2D(64, 3, strides=1, padding="same", 
                      kernel_initializer=init,
                      name=layer_name(prefix,"Conv",7))(x)
    x = layers.BatchNormalization(name=layer_name(prefix,"BNorm",7))(x)
    x = layers.ReLU(name=layer_name(prefix,"Relu",7))(x)
    x = layers.Conv2D(64, 4, strides=2, padding="same", 
                      kernel_initializer=init,
                      name=layer_name(prefix,"Conv",8))(x)
    x = layers.BatchNormalization(name=layer_name(prefix,"BNorm",8))(x)
    det_features = layers.ReLU(name=layer_name(prefix,"Relu",8))(x)
    # 16x16
    """
    
    # 32x32
    return inputs, det_features, mid_features


"""
CHANGELOG 2021/03/04: 
    Use global average pooling to summarize feature maps,
    then use dense layer to weigh all feature maps.
    Class activation heatmaps: feature map-wise sum of pre-GAP feature maps 
        multiplied by their corresponding connection weights to post-GAP dense layer
    (Zhou et al., 2015: Learning Deep Features for Discriminative Localization)
CHANGELOG 2021/03/12:
    Use traditional convolutional neural network structure for DCT variants.
"""


def build_gap_decision_layer(features, init, prefix, softmax=False, num_classes=1):
    """
    Assembles a decision layer with the class activation mapping topology.
    Parameters
    ----------
    features : keras.layers.Layer
        Extracted features from previous layers.
    init : keras.initializers.Initializer
        Weight initializer.
    prefix : str
        Prefix added to all layer names.
    softmax : bool, optional
        Softmax for multiclass outputs. False by default.
    num_classes : int, optional
        Number of classes specified for multiclass outputs. 
        The default is 1. Only considered if softmax=True.
    Returns
    -------
    prediction : keras.layers.Layer
        Decision layer output(s).
    """
    x = layers.GlobalAveragePooling2D(name=f"{prefix}GAP")(features)
    if softmax:
        prediction = layers.Dense(num_classes, activation="softmax", kernel_initializer=init,
                                  name=f"{prefix}Decision")(x)
    else:
        prediction = layers.Dense(1, activation="sigmoid", kernel_initializer=init, 
                                  name=f"{prefix}Decision")(x)  
    return prediction


def build_flat_decision_layer(features, init, prefix, softmax=False, num_classes=1):
    """
    Assembles a conventional, fully-connected decision layer.
    Parameters
    ----------
    features : keras.layers.Layer
        Extracted features from previous layers.
    init : keras.initializers.Initializer
        Weight initializer.
    prefix : str
        Prefix added to all layer names.
    softmax : bool, optional
        Softmax for multiclass outputs. False by default.
    num_classes : int, optional
        Number of classes specified for multiclass outputs. 
        The default is 1. Only considered if softmax=True.
    Returns
    -------
    prediction : keras.layers.Layer
        Decision layer output(s).
    """
    x = layers.Flatten(name=f"{prefix}Flatten")(features)
    if softmax:
        prediction = layers.Dense(num_classes, activation="softmax", kernel_initializer=init,
                                  name=f"{prefix}Decision")(x)
    else:
        prediction = layers.Dense(1, activation="sigmoid", kernel_initializer=init, 
                                  name=f"{prefix}Decision")(x)  
    return prediction
    

def build_secondary_layers(det_features, init, num_fmaps=32, prefix="a"):
    
    """
    Assembles secondary convolutional module, used for source generator attribution.
    DOES NOT CONTAIN AN INPUT LAYER OR PRODUCE A FULL NEURAL NET!
    
    Parameters
    ----------
    det_features : keras.layers.Layer
        Intermediate feature extraction outputs from primary layers.
    init : keras.initializers.Initializer
        Weight initializer.
    num_fmaps : int
        Number of feature maps per layer. The default is 32.
    prefix : str, optional
        Prefix added to all layer names. The default is "a".

    Returns
    -------
    att_features: keras.layers.Layer
        Complete secondary feature extraction outputs.
        
    """
    
    x = layers.Conv2D(num_fmaps, 3, strides=1, padding="same", 
                      kernel_initializer=init,
                      name=layer_name(prefix,"Conv",1))(det_features)
    x = layers.BatchNormalization(name=layer_name(prefix,"BNorm",1))(x)
    x = layers.ReLU(name=layer_name(prefix,"Relu",1))(x)
    x = layers.Conv2D(num_fmaps, 4, strides=2, padding="same", 
                      kernel_initializer=init,
                      name=layer_name(prefix,"Conv",2))(x)
    x = layers.BatchNormalization(name=layer_name(prefix,"BNorm",2))(x)
    att_features = layers.ReLU(name=layer_name(prefix,"Relu",2))(x)
    
    return att_features


def build_primary_model(input_shape, init=None, gap=True, cam=False, 
                        prefix="d", softmax=False, num_classes=1):
    
    """
    Returns a model containing only the primary module.
    
    Parameters
    ----------
    input_shape : (*int)
        Input image tensor shape.
    init : keras.initializers.Initializer, optional
        Weight initializer.
    gap : bool, optional
        Use the class activation mapping topology. The default is True.
    cam : bool, optional
        Include feature maps with model outputs for class activation mapping.
        Set to True for visualizing model behaviour. The default is False.
    softmax : bool, optional
        Softmax for multiclass outputs. False by default.
    num_classes : int, optional
        Number of classes specified for multiclass outputs. 
        The default is 1. Only considered if softmax=True.
        
    """
    
    if init is None:
        init = initializers.RandomNormal(stddev=0.02)
        
    inputs, det_layers, _ = build_primary_layers(input_shape, init)
    if gap:
        full_layers = build_gap_decision_layer(det_layers, init, prefix=prefix, 
                                               softmax=softmax, num_classes=num_classes)
    else:
        full_layers = build_flat_decision_layer(det_layers, init, prefix=prefix,
                                                softmax=softmax, num_classes=num_classes)
        
    if cam:
        model = keras.Model(inputs, [full_layers, det_layers])
    else:
        model = keras.Model(inputs, [full_layers])
        
    return model


def build_full_model(input_shape, init=None, gap=True, cam=False):
    
    """
    Returns a model containing the primary module and one secondary module.
    
    Parameters
    ----------
    input_shape : (*int)
        Input image tensor shape.
    init : keras.initializers.Initializer, optional
        Weight initializer.
    gap : bool, optional
        Use the class activation mapping topology. The default is True.
    cam : bool, optional
        Include feature maps with model outputs for class activation mapping.
        Set to True for visualizing model behaviour. The default is False.
        
    """
    
    if init is None:
        init = initializers.RandomNormal(stddev=0.02)
        
    inputs, det_layers, mid_layers = build_primary_layers(input_shape, init)
    att_layers = build_secondary_layers(mid_layers, init)
    
    if gap:
        full_det = build_gap_decision_layer(det_layers, init, prefix="d")
        full_att = build_gap_decision_layer(att_layers, init, prefix="a")
    else:
        full_det = build_flat_decision_layer(det_layers, init, prefix="d")
        full_att = build_flat_decision_layer(att_layers, init, prefix="a")
        
    if cam:
        model = keras.Model(inputs, [full_det, det_layers, full_att, att_layers])
    else:
        model = keras.Model(inputs, [full_det, full_att])
        
    return model


def build_multilabel_model(input_shape, num_classes=1, init=None, gap=True, cam=False):
    
    """
    Returns a deepfake detection and multi-class binary source generator attribution model.
    Many independent secondary layers will be included to extract traces from different sources.
    
    Parameters
    ----------
    input_shape : (*int)
        Input image tensor shape.
    num_sources : int, optional
        Number of source generators (and therefore branched secondary layers) to consider.
        The default is 1.
    init : keras.initializers.Initializer, optional
        Weight initializer.
    gap : bool, optional
        Use the class activation mapping topology. The default is True.
    cam : bool, optional
        Include feature maps with model outputs for class activation mapping.
        Set to True for visualizing model behaviour. The default is False.
        
    """
    
    if init is None:
        init = initializers.RandomNormal(stddev=0.02)
        
    inputs, det_layers, mid_layers = build_primary_layers(input_shape, init)
    if gap:
        full_det = build_gap_decision_layer(det_layers, init, prefix="d")
    else:
        full_det = build_flat_decision_layer(det_layers, init, prefix="d")
        
    att_layer_list = []
    for i in range(num_classes):
        att_layers = build_secondary_layers(mid_layers, init, prefix=f"a{i+1}")
        if gap:
            full_att = build_gap_decision_layer(att_layers, init, prefix=f"a{i+1}")
        else:
            full_att = build_flat_decision_layer(att_layers, init, prefix=f"a{i+1}")
        att_layer_list.append(full_att)
        if cam:
            att_layer_list.append(att_layers)
        
    if cam:
        output_list = [full_det, det_layers]
    else:
        output_list = [full_det]
    output_list.extend(att_layer_list)
    model = keras.Model(inputs, output_list)
    
    return model
    
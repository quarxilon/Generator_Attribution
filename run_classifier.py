"""
Adapted from original code by Frank et al., 2020:
https://github.com/RUB-SysSec/GANDCTAnalysis
"""

import argparse
import cv2
import os

import numpy as np
import tensorflow as tf
import datetime as dt

from tqdm import tqdm
from pathlib import Path
from tensorflow import keras

from data_prep.maths import log_scale
from data_prep.dataset_util import image_paths
from data_prep.img_numpy import load_image, dct2
from data_prep.img_tflow import dct2_tf
from classifier.models import (build_detection_model, 
                               build_attribution_model)

# global default parameters
BATCH_SIZE = 32
INPUT_SHAPE = (256, 256, 3)


class DCTLayer(keras.layers.Layer):
    """
    Defines a raw image preprocessing layer for DCT input classifiers.
    Dataset mean and standard deviation must be provided during instantiation for normalization.
    """
    
    def __init__(self, mean, std):
        super(DCTLayer, self).__init__()
        self.mean = mean
        self.std = std

    def build(self, input_shape):
        self.mean_w = self.add_weight(
            "mean", shape=input_shape[1:], initializer=keras.initializers.Constant(self.mean), trainable=False)
        self.std_w = self.add_weight(
            "std", shape=input_shape[1:], initializer=keras.initializers.Constant(self.std), trainable=False)

    def call(self, inputs):
        # apply DCT
        x = dct2_tf(inputs)
        # log scale
        x = tf.abs(x)
        x += 1e-12
        x = tf.math.log(x)
        # remove mean + unit variance
        x = x - self.mean_w
        x = x / self.std_w
        return x


class PixelLayer(keras.layers.Layer):
    """
    Defines a raw image preprocessing layer for pixel (RGB) input classifiers.
    """
    
    def __init__(self):
        super(PixelLayer, self).__init__()

    def call(self, inputs):
        # normalize
        x = (inputs / 127.5) - 1.0
        return x
    
    
def init_tf(args):
    """
    TensorFlow configuration script.
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    print("TensorFlow initialized...")


def _pixel_model(model_type, input_shape, model_path):
    
    """
    Constructs and loads a pixel (RGB) input model.
    
    Parameters
    ----------
    model_type : string
        Type of classifier: deepfake detection only ("det") or with attribution ("att").
    input_shape : (*int)
        Input image tensor shape.
    model_path : string
        Trained model weights filepath.

    Returns
    -------
    model : keras.Model
        Loaded model.
    det_cam_weights : keras.layer.Layer.weights
        Primary decision layer weights (for CAM)
    att_cam_weights : keras.layer.Layer.weights
        Secondary decision layer weights (for CAM)
        
    """
    
    if model_type == "att":
        model = build_attribution_model(input_shape, gap=True, cam=True)
    else:
        model = build_detection_model(input_shape, gap=True, cam=True)
        
    # Loads model weights and disables trainability
    model.load_weights(model_path, by_name=True)
    for layer in model.layers:
        layer.trainable = False
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.training = False
    model.summary()
        
    # Extract weights from decision layer(s) for class activation mapping
    det_cam_weights = model.get_layer(name="dDecision").get_weights()[0]
    if model_type == "att":
        att_cam_weights = model.get_layer(name="aDecision").get_weights()[0]
    else:
        att_cam_weights = None
    
    inputs = keras.Input(shape=input_shape)
    prep_layer = PixelLayer()(inputs)
    new_outputs = model(prep_layer)
    model = keras.Model(inputs, new_outputs)
    # Print model summary
    model.summary()
    
    return model, det_cam_weights, att_cam_weights


def _dct_model(model_type, input_shape, model_path, mean, std):
    
    """
    Constructs and loads a DCT input model.
    
    Parameters
    ----------
    model_type : string
        Type of classifier: deepfake detection only ("det") or with attribution ("att").
    input_shape : (*int)
        Input image tensor shape.
    model_path : string
        Trained model weights filepath.
    mean : np.array
        Dataset sample mean for normalization.
    std : np.array
        Dataset sample standard deviation for normalization.

    Returns
    -------
    model : keras.Model
        Loaded model.
        
    """
    
    if model_type == "att":
        model = build_attribution_model(input_shape, gap=False, cam=True)
    else:
        model = build_detection_model(input_shape, gap=False, cam=True)
    
    # Loads model weights and disables trainability
    model.load_weights(model_path, by_name=True)
    for layer in model.layers:
        layer.trainable = False
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.training = False
    model.summary()
    
    inputs = keras.Input(shape=input_shape)
    prep_layer = DCTLayer(mean, std)(inputs)
    new_outputs = model(prep_layer)
    model = keras.Model(inputs, new_outputs)
    # Print model summary
    model.summary()
    
    return model


def _process_image(path):
    """
    Helper function to load RGB images for inference.
    """
    image = load_image(path, greyscale=False, tf=True)
    image = image.astype(np.float32)
    return image


def _process_image_gs(path):
    """
    Helper function to load greyscale images for inference.
    """
    image = load_image(path, greyscale=True, tf=True)
    image = image.astype(np.float32)
    return image


def _process_image_cv2(path):
    """
    Alias for cv2.imread.
    """
    return cv2.imread(path)


def _convert_cv2_dct(pixels):
    """
    Converts an OpenCV raw image into its DCT spectrum.
    """
    image = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    image = dct2(image)
    image = log_scale(image)
    image = np.uint8(np.clip((255 * image), 0, 255))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _load_images(path, amount=None, batch_size=BATCH_SIZE, greyscale=False, sourcetags=False):
    
    """
    Load images into classifier.
    
    Parameters
    ----------
    path : string
        Filepath of the input data directory.
    amount : int, optional
        Number of images in the input data directory to classify.
    batch_size : int, optional
        Batch size. The default is 32.
        Throws ValueError if batch size exceeds specified amount.
    greyscale : bool, optional
        Whether to load images in greyscale. The default is False.
    sourcetags : bool, optional
        Whether to prefix image IDs with their source labels. The default is False.

    Returns
    -------
    images : [np.array, [string]]
        List of images and their filepaths.
    paths : [string]
        List of image filepaths.
    image_ids : [string]
        List of image IDs.
        
    """
    
    paths = []
    image_ids = []
    img_dirs = list(map(str, filter(lambda x: x.is_dir(), Path(path).iterdir())))
    for img_dir in img_dirs:
        raw_paths = image_paths(img_dir)
        image_ids.extend(raw_paths)
        paths.extend(list(map(str, raw_paths)))
        
    image_id_format = (lambda x: f"{x.parent.stem}_{x.stem}") if sourcetags else (lambda x: x.stem)
    image_ids = list(map(image_id_format, image_ids))
    
    if len(paths) < batch_size:
        raise ValueError("Batch size exceeds available data! Please reduce the batch size.")
    if amount is not None:
        paths = paths[:amount]
        image_ids = image_ids[:amount]
        
    actual_amount = len(paths) - (len(paths) % batch_size)
    if actual_amount < len(paths):
        paths = paths[:actual_amount]
        image_ids = image_ids[:actual_amount]
        
    images = list(map(_process_image_gs if greyscale else _process_image, paths))
        
    return images, paths, image_ids


def _impose_heatmap(raw_heatmap, image, cmap=cv2.COLORMAP_JET):
    
    """
    Superimpose activation heatmap on classified images.
    
    Parameters
    ----------
    raw_heatmap : np.array
        Activation heatmap.
    image : np.array
        RGB image or DCT spectrum.
    cmap
        OpenCV colour map.

    """
    
    heatmap = cv2.resize(raw_heatmap, image.shape[0:2])
    heatmap = cv2.applyColorMap(np.uint8(np.clip((255 * heatmap), 0, 255)), cmap)
    return cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)


def gradcam_dct(raw_images, fmaps, gradients, cmap=cv2.COLORMAP_JET):
    
    """
    GradCAM implementation for DCT input classifiers.
    Returns a list of activation heatmaps.
    
    Parameters
    ----------
    raw_images : [np.array]
        RGB image.
    fmaps
        Feature maps from final convolutional layer.
    gradients
        Predicted class-specific gradients for GradCAM.
    cmap
        OpenCV colour map.

    """
    
    dcts = list(map(_convert_cv2_dct, raw_images))
    gradcams = tf.math.multiply(gradients, fmaps)
    gradcams = tf.reduce_sum(gradcams, axis=3)
    gradcams = tf.nn.relu(gradcams)
    gradcams = gradcams / tf.math.reduce_max(gradcams)
    gradcams = gradcams.numpy()
    return list(map(_impose_heatmap, gradcams, dcts, cmap))





def main(args):
    
    if args.MODEL_TYPE not in ("det", "att"):
        raise NotImplementedError("Invalid model type specified!")
    init_tf(args)
        
    current_timestamp = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_name = Path(args.MODEL).stem
    output_path = Path(args.OUTPUT.rstrip('/')).joinpath(f"{model_name}_{current_timestamp}")
    cv2_font = cv2.FONT_HERSHEY_SIMPLEX
    threshold = 0.5
    
    images, image_paths, image_ids = _load_images(args.DATA, amount=args.size, batch_size=BATCH_SIZE,
                                                  greyscale=(True if (INPUT_SHAPE[2] == 1) else False),
                                                  sourcetags=args.sourcetags)
        
    if args.dct:
        model = _dct_model(args.MODEL_TYPE, INPUT_SHAPE, args.MODEL,
                           np.load(f"{args.normstats.rstrip('/')}/mean.npy"),
                           np.load(f"{args.normstats.rstrip('/')}/std_dev.npy"))
    else:
        model, det_cam_weights, att_cam_weights = _pixel_model(
            args.MODEL_TYPE, INPUT_SHAPE, args.MODEL)
        
    # counters
    fake_count = 0
    if args.MODEL_TYPE == "att":
        att_count = 0
        stupid_count = 0
    
    # BATCH-WISE LOOP
    for i in tqdm(range(0, len(images), BATCH_SIZE)):
        
        with tf.GradientTape(persistent=True) as tape:
            predictions = model(np.asarray(images[i: i + BATCH_SIZE]))
            det_preds = predictions[0]
            det_preds = tf.reshape(det_preds, shape=(-1,))
            det_fmaps = predictions[1]
            if args.MODEL_TYPE == "att":
                att_preds = predictions[2]
                att_preds = tf.reshape(att_preds, shape=(-1,))
                att_fmaps = predictions[3]
        
        batch_output_path = str(output_path.joinpath(f"{i + 1}_{i + BATCH_SIZE}").absolute())
        if not os.path.exists(batch_output_path):
            os.makedirs(batch_output_path)
        real_ids = []
        img_ids = image_ids[i: i + BATCH_SIZE]
        
        # Deepfake detection
        det_images = list(map(_process_image_cv2, image_paths[i: i + BATCH_SIZE]))
        if args.MODEL_TYPE == "att":
            att_images = det_images.copy()
        
        if args.dct: # GradCAM
            det_grads = tape.gradient(det_preds, det_fmaps)
            det_dcts = gradcam_dct(det_images.copy(), det_fmaps, det_grads)
        else: # CAM
            det_cams = np.zeros(shape=det_fmaps.shape[0:3], dtype=np.float32)
            for j in range(BATCH_SIZE):
                for k, w in enumerate(det_cam_weights[:, 0]):
                    det_cams[j, :, :] += w * det_fmaps.numpy()[j, :, :, k]
            det_images = list(map(_impose_heatmap, det_cams, det_images))
        
        for j, pred in enumerate(det_preds):
            if pred.numpy() > threshold:
                fake_count += 1
                label_text = f"FAKE:{pred.numpy(): 3.2%}"
                label_bgr = (255,255,255)
            else:
                real_ids.append(j)
                label_text = f"REAL:{(1.00 - pred.numpy()): 3.2%}"
                label_bgr = (255,255,255) if args.dct else (0,255,0)
            if not args.nolabels:
                if args.dct:
                    det_dcts[j] = cv2.putText(det_dcts[j], label_text, (8, 16), cv2_font, 
                                              0.5, label_bgr, 1, cv2.LINE_AA)
                else:
                    det_images[j] = cv2.putText(det_images[j], label_text, (8, 16), cv2_font, 
                                                0.5, label_bgr, 1, cv2.LINE_AA)
        if args.dct:
            det_images = [cv2.hconcat([det_images[j], det_dcts[j]]) for j in range(BATCH_SIZE)]
                
        # Image source attribution
        if args.MODEL_TYPE == "att":
            
            if args.dct: # GradCAM
                att_grads = tape.gradient(att_preds, att_fmaps)
                att_dcts = gradcam_dct(att_images, att_fmaps, att_grads)
            else: # CAM
                att_cams = np.zeros(shape=att_fmaps.shape[0:3], dtype=np.float32)
                for j in range(BATCH_SIZE):
                    for k, w in enumerate(att_cam_weights[:, 0]):
                        att_cams[j, :, :] += w * att_fmaps.numpy()[j, :, :, k]
                att_images = list(map(_impose_heatmap, att_cams, att_images))
            
            for j, pred in enumerate(att_preds):
                if pred.numpy() > threshold:
                    att_count += 1
                    if j in real_ids:
                        stupid_count += 1
                if not args.nolabels:
                    label_text = f"{args.source}:"
                    lower_text = f"{pred.numpy(): <3.2%}"
                    label_bgr = (255,255,255)
                    if args.dct:
                        att_dcts[j] = cv2.putText(att_dcts[j], label_text, (8, 16), cv2_font,
                                                  0.5, label_bgr, 1, cv2.LINE_AA)
                        att_dcts[j] = cv2.putText(att_dcts[j], lower_text, (8, 32), cv2_font,
                                                  0.5, label_bgr, 1, cv2.LINE_AA)
                    else:
                        att_images[j] = cv2.putText(att_images[j], label_text, (8, 16), cv2_font,
                                                    0.5, label_bgr, 1, cv2.LINE_AA)   
                        att_images[j] = cv2.putText(att_images[j], lower_text, (8, 32), cv2_font,
                                                    0.5, label_bgr, 1, cv2.LINE_AA)
            if args.dct:
                det_images = [cv2.hconcat([det_images[j], att_dcts[j]]) for j in range(BATCH_SIZE)]
            else:
                det_images = [cv2.hconcat([det_images[j], att_images[j]]) for j in range(BATCH_SIZE)]
                    
        for j in range(BATCH_SIZE):
            cv2.imwrite(f"{batch_output_path}/{img_ids[j]}.png", det_images[j])
        del tape
        # END OF BATCH-WISE LOOP

    print(f"{fake_count/len(images): 3.2%} ({fake_count}) of the {len(images)} assessed images are predicted to be of GAN origin.")
    if args.MODEL_TYPE == "att":
        print(f"{att_count/len(images): 3.2%} ({att_count}) of images are predicted to have been generated by {args.source}.")
        if stupid_count > 0:
            print(f"WARNING: {stupid_count/att_count: 3.2%} ({stupid_count}) of images attributed to {args.source} were predicted to be real.")


def parse_args():
    
    global BATCH_SIZE, SIZE, INPUT_SHAPE
    parser = argparse.ArgumentParser()

    parser.add_argument("MODEL_TYPE",           help="Select classifier type {det, att}.", type=str)
    parser.add_argument("MODEL",                help="Path to trained model.", type=str)
    parser.add_argument("DATA",                 help="Directory of images to classify.")
    parser.add_argument("OUTPUT",               help="Directory of labelled image outputs")
    parser.add_argument("--dct",        "-f",   help="Accept DCT coefficients as input", action="store_true")
    parser.add_argument("--normstats",  "-n",   help="Directory of mean/var/stdev for log-norm DCT coefficients.", type=str, default=None)
    parser.add_argument("--size",       "-s",   help="Only process this amount of images.", type=int, default=None)
    parser.add_argument("--batch_size", "-b",   help=f"Batch size. Default: {BATCH_SIZE}.", type=int, default=None)
    parser.add_argument("--image_size", "-i",   help=f"Image size. Default: {INPUT_SHAPE[0]}", type=int, default=256)
    parser.add_argument("--source",     "-l",   help="Designated source label for attribution.", type=str, default="[SOURCE]")
    parser.add_argument("--greyscale",  "-g",   help="Train on greyscale images.", action="store_true")
    parser.add_argument("--nolabels",   "-x",	help="Exclude predicted labels from output heatmaps.", action="store_true")
    parser.add_argument("--sourcetags", "-t",	help="Add source label prefixes to image IDs; required for some datasets.", action="store_true")

    args = parser.parse_args()
    
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.greyscale or args.dct:
        INPUT_SHAPE = (args.image_size, args.image_size, 1)
    else:
        INPUT_SHAPE = (args.image_size, args.image_size, 3)

    return args


if __name__ == "__main__":
    main(parse_args())
    
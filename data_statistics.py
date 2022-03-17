import argparse
import functools
import numpy as np
import os

from pathlib import Path
from data_prep.dataset_util import image_paths
from data_prep.img_numpy import load_image, dct2
from data_prep.maths import log_scale, welford


"""
Calculates and saves dataset statistics.
Currently available features: dctnorm
"""


def _dct2_wrapper(image, log=False):
    image = dct2(np.asarray(image))
    return log_scale(image) if log else image


def _get_image_id(image_path):
    if type(image_path) is not Path:
        image_path = Path(image_path)
    image_id = image_path.stem.split('_')[0]
    if not image_id.isnumeric():
        image_id = image_path.stem.split('_')[1]
    assert image_id.isnumeric()
    return int(image_id)


def _collect_image_paths(directory, input_size):
    images = list(sorted(image_paths(directory), key=_get_image_id))
    assert len(images) >= input_size, \
        f"input_size ({input_size}) exceeds no. of images ({len(images)})!"
    dataset = images[:input_size]
    assert len(dataset) == input_size
    return dataset


def collect_all_paths(pos_dirs, neg_dirs, input_size_pos, input_size_neg):
    """
    Collects image filepaths from every specified source directory, 
    up to the first input_size images per source.
    """
    pos_directories = sorted(map(str, filter(
        lambda x: x.is_dir(), Path(pos_dirs).iterdir())))
    neg_directories = sorted(map(str, filter(
        lambda x: x.is_dir(), Path(neg_dirs).iterdir())))

    dataset = []
    
    for directory in pos_directories:
        dataset.extend(_collect_image_paths(directory, input_size_pos))
        
    for directory in neg_directories:
        dataset.extend(_collect_image_paths(directory, input_size_neg))
        
    dataset = np.asarray(dataset)
        
    return dataset


def main(args):
    load_function = functools.partial(load_image, greyscale=True)
    transformation_function = None
	
    if not os.path.exists(args.OUT_DIRECTORY.rstrip('/')):
        os.makedirs(args.OUT_DIRECTORY.rstrip('/'))
    
    if args.dctnorm:
        transformation_function = functools.partial(_dct2_wrapper, log=True)
        images = collect_all_paths(args.POS_DIRECTORY, 
                                   args.NEG_DIRECTORY, 
                                   args.input_size_pos,
                                   args.input_size_neg)
        images = map(load_function, images)
        images = map(transformation_function, images)

        mean, variance = welford(images)
        std_dev = np.sqrt(variance)
        
        with open(f"{args.OUT_DIRECTORY.rstrip('/')}/mean.npy", "wb") as file:
            print(f"MEAN: {mean}")
            np.save(file, mean)
        with open(f"{args.OUT_DIRECTORY.rstrip('/')}/variance.npy", "wb") as file:
            print(f"VARIANCE: {variance}")
            np.save(file, variance)
        with open(f"{args.OUT_DIRECTORY.rstrip('/')}/std_dev.npy", "wb") as file:
            print(f"STD DEV: {std_dev}")
            np.save(file, std_dev)


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "POS_DIRECTORY", help="Directory of dataset (positive portion).", type=str)
    parser.add_argument(
        "NEG_DIRECTORY", help="Directory of dataset (negative portion).", type=str)
    parser.add_argument(
        "OUT_DIRECTORY", help="Directory to save output statistics.", type=str)
    parser.add_argument(
        "--input_size_pos", "-p", help="Number of images per positively labelled source to consider. Default: 7000", type=int, default=7000)
    parser.add_argument(
        "--input_size_neg", "-n", help="Number of images per negatively labelled source to consider. Default: 7000", type=int, default=7000)
    parser.add_argument(
        "--dctnorm", "-d", help="Output mean, var, std of DCT spectra via welford method.", action="store_true")
    
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
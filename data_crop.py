"""
Script for cropping CelebA and LSUN adapted from: 
https://github.com/ningyu1991/GANFingerprints/
https://github.com/RUB-SysSec/GANDCTAnalysis
"""

import argparse

import numpy as np

from PIL import Image
from pathlib import Path

from skimage.transform import resize
from concurrent.futures import ProcessPoolExecutor
from data_prep.dataset_util import image_paths


"""
Script to resize image datasets and crop images into squares where necessary.
"""


# global parameters
RESIZE_RES = [128, 128]
AMOUNT = 10000


def _get_image_id(image_path):
    """
    Extracts the image sample ID (index) from an image filepath as an integer.
    """
    if type(image_path) is not Path:
        image_path = Path(image_path)
    i = 0
    image_id = image_path.stem.split('_')[i]
    while not image_id.isnumeric():
        i += 1
        image_id = image_path.stem.split('_')[i]
    return int(image_id)


def transform_image(image_path, output_dir, celeba_flag=False, jpeg_flag=False):
    
    """
    Applies specified transformations to images and then saves them.
    """
    
    image = np.asarray(Image.open(image_path))

    if image.shape[0] != RESIZE_RES[0] or image.shape[1] != RESIZE_RES[1]:
        x, y, _ = image.shape
        
        if celeba_flag:
            # CelebA cropping procedure
            image = np.copy(image)
            x_upper = min(121+64, x)
            y_upper = min(89+64, y)
            image = image[x_upper-128:x_upper, y_upper-128:y_upper]
            
        else:
            # center crop towards smaller side
            if x < y:
                y_center = y // 2
                crop = (y - x) // 2
                image = np.copy(image)
                image = image[:, y_center-crop:y_center+crop]

            elif x > y:
                x_center = x // 2
                crop = (x - y) // 2
                image = np.copy(image)
                image = image[x_center-crop:x_center+crop, :]
            
            # resize images using scikit-image
            image = resize(image.astype(np.float64), RESIZE_RES)
        
        image = np.clip(image, 0, 255.).astype(np.uint8)

    if jpeg_flag:
        Image.fromarray(image).save(f"{output_dir}/{image_path.stem}.jpg")
    else:
        Image.fromarray(image).save(f"{output_dir}/{image_path.stem}.png")


def main(args):
    global RESIZE_RES, AMOUNT
    if args.resolution != 128:
        RESIZE_RES = [args.resolution, args.resolution]
        if args.celeba:
            raise NotImplementedError("Only 128x128 supported for CelebA!")
    if args.amount != AMOUNT:
        AMOUNT = args.amount
    Path(args.OUTPUT.rstrip('/')).mkdir(exist_ok=True)
    images = list(sorted(image_paths(args.SOURCE), key=_get_image_id))[:AMOUNT]
    images_metadata = map(lambda i: (Path(i), args.OUTPUT, args.celeba, args.jpeg_output), images)
    with ProcessPoolExecutor() as pool:
        list(pool.map(transform_image, *images_metadata))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("SOURCE",               help="Source directory containing raw images of a certain class.", type=str)
    parser.add_argument("OUTPUT",               help="Output directory containing cropped and/or resized images.", type=str)
    parser.add_argument("--resolution", "-r",   help="Set resolution of resized images. Default: 128", type=int, default=128)
    parser.add_argument("--amount", "-a",       help="Amount of images to crop and/or resize. Default: 10,000", type=int, default=10000)
    parser.add_argument("--celeba", "-c",       help="CelebA cropping mode (for GANFP dataset)", action="store_true")
    parser.add_argument("--jpeg_output", "-j",  help="Output images as .jpg instead of .png", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
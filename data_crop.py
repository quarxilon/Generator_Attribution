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
RESIZE_RES = (128, 128)
AMOUNT = 10000
CELEBA_FLAG = False


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


def transform_image(image_path):
    
    """
    Applies transformations to an image.
    """
    
    image = np.asarray(Image.open(image_path))

    if image.shape[0] != RESIZE_RES[0] or image.shape[1] != RESIZE_RES[1]:
        row, col, _ = image.shape
        
        if CELEBA_FLAG:
            # CelebA cropping procedure
            # original image size 178x218
            image = np.copy(image)
            row_upper = min(121+64, row)
            col_upper = min(89+64, col)
            image = image[row_upper-128:row_upper, col_upper-128:col_upper]
            
        else:
            # center crop towards smaller side
            if row < col: # landscape to square
                crop = (col - row) // 2
                image = np.copy(image)
                image = image[:, crop:col-crop]

            elif row > col: # portrait to square
                crop = (row - col) // 2
                image = np.copy(image)
                image = image[crop:row-crop, :]
            
            # resize images using scikit-image
            if image.shape != RESIZE_RES:
                image = resize(image.astype(np.float64), RESIZE_RES)
        
        image = np.clip(image, 0, 255.).astype(np.uint8)
        
    return image, image_path


def main(args):
    global RESIZE_RES, AMOUNT, CELEBA_FLAG
    if args.resolution != 128:
        if args.celeba:
            raise NotImplementedError("Desired output resolution not supported!")
        RESIZE_RES = [args.resolution, args.resolution]
    if args.amount != AMOUNT:
        AMOUNT = args.amount
    CELEBA_FLAG = True if args.celeba else False
    Path(args.OUTPUT.rstrip('/')).mkdir(exist_ok=True, parents=True)
    images = list(sorted(image_paths(args.SOURCE), key=_get_image_id))[:AMOUNT]
    with ProcessPoolExecutor() as pool:
        for img_arr, img_path in pool.map(transform_image, images):
            if args.jpeg_output:
                Image.fromarray(img_arr).save(f"{args.OUTPUT}/{img_path.stem}.jpg")
            else:
                Image.fromarray(img_arr).save(f"{args.OUTPUT}/{img_path.stem}.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("SOURCE",               help="Source directory containing raw images of a certain class.", type=str)
    parser.add_argument("OUTPUT",               help="Output directory containing cropped and/or resized images.", type=str)
    parser.add_argument("--resolution", "-r",   help="Set resolution of resized images. Default: 128", type=int, default=128)
    parser.add_argument("--amount", "-a",       help="Amount of images to crop and/or resize. Default: 10,000", type=int, default=10000)
    parser.add_argument("--celeba", "-c",       help="CelebA-specific cropping mode (for GAN Fingerprints dataset only)", action="store_true")
    parser.add_argument("--jpeg_output", "-j",  help="Output images as .jpg instead of .png", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
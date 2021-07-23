"""
Adapted from original code by Frank et al., 2020:
https://github.com/RUB-SysSec/GANDCTAnalysis
"""

import argparse
import random
import cv2
import numpy as np

from PIL import Image
from pathlib import Path
from data_prep.dataset_util import image_paths


"""
Script to introduce "laundering" perturbations into image datasets as augmentations.
If multimodal augmentation is specified, augmentations will be performed in the following order:
    BLUR, CROP, JPEG, NOISE
"""


# probability of each image receiving an individual augmentation
AUGMENT_PROB = 0.5


def noise(image):
    """
    Variance uniformly sampled from U[5.0,20.0]
    """
    variance = np.random.uniform(low=5.0, high=20.0)
    image = np.copy(image).astype(np.float64)
    noise = variance * np.random.randn(*image.shape)
    image += noise
    return np.clip(image, 0.0, 255.0).astype(np.uint8), variance


def blur(image):
    """
    Kernel size sampled from [3, 5, 7, 9]
    """
    kernel_size = np.random.choice([3, 5, 7, 9])
    return cv2.GaussianBlur(
        image, (kernel_size, kernel_size), sigmaX=cv2.BORDER_DEFAULT), kernel_size


def jpeg(image):
    """
    Compression quality factor uniformly sampled from U[10, 75]
    """
    factor = np.random.randint(low=10, high=75)
    _, image = cv2.imencode(".jpg", image, [factor, 90])
    return cv2.imdecode(image, 1), factor


def crop(image,symmetric=False):
    """
    Asymmetrically crop between 5% to 20% of the image
    """
    percentage = np.random.uniform(low=0.05, high=0.2)
    if symmetric:
        x_dist, y_dist = [0.5, 0.5]
    else:
        x_dist, y_dist = np.random.sample(2)
    
    x, y, _ = image.shape
    x_crop = int(x * percentage)
    y_crop = int(y * percentage)
    
    x_crop_left = int(x_crop * x_dist)
    x_crop_right = x_crop - x_crop_left
    y_crop_top = int(y_crop * y_dist)
    y_crop_bottom = y_crop - y_crop_top
    
    cropped = image[x_crop_left: -x_crop_right, y_crop_top: -y_crop_bottom]
    resized = cv2.resize(cropped, dsize=(x, y), interpolation=cv2.INTER_CUBIC)
    return resized, percentage*100


def apply_transformation_to_dataset(in_dir, out_dir, mode, size=None, jpg_flag=False):
    
    """
    Applies specified augmentations to the dataset and saves it.
    
    Parameters
    ----------
    in_dir : str
        Filepath to clean image directory.
    out_dir : str
        Filepath to output directory.
    mode : str
        Type of augmentation to apply.
    size : int, optional
        Maximum number of images per class to process.
    jpg_flag : bool, optional
        Whether to save images as .jpg instead of .png. The default is False.
        
    """
    
    if mode == "blur":
        image_functions = [blur]
    elif mode == "crop":
        image_functions = [crop]
    elif mode == "jpeg":
        image_functions = [jpeg]
    elif mode == "noise":
        image_functions = [noise]
    elif mode == "multi":
        image_functions = [blur, crop, jpeg, noise]
    else:
        raise NotImplementedError("Selected unrecognized mode: {mode}!")
        
    out_dir_path = Path(out_dir.rstrip('/'))

    for class_dir_path in list(map(str, filter(
            lambda x: x.is_dir(), Path(in_dir.rstrip('/')).iterdir()))):
        class_out_path = out_dir_path.joinpath(f"{class_dir_path.stem}_{mode}")
        class_out_path.mkdir(exist_ok=True, parents=True)
        if size is None:
            images = image_paths(class_dir_path)
        else:
            images = image_paths(class_dir_path)[:size]
        # images = map(np.asarray, map(Image.open, images))
        
        num_augmented = 0
        for image in images:
            new_image = np.asarray(Image.open(image))
            suffix = str()
            
            is_augmented = False
            for image_function in image_functions:
                if np.random.sample() >= AUGMENT_PROB:
                    aug_image, param = image_function(new_image)
                    # assert that image and aug_image are not entirely identical
                    assert not np.isclose(aug_image, new_image).all()
                    new_image = aug_image
                    suffix += f"_{image_function.__name__}{int(param)}"
                    if not is_augmented:
                        num_augmented += 1
                        is_augmented = True

            Image.fromarray(new_image).save(class_out_path.joinpath(
                f"{image.stem}{suffix}").with_suffix('.jpg' if jpg_flag else '.png').__str__())
            if is_augmented:
                print(f"\rConverted {num_augmented:06} out of {len(images) if size is None else max(len(images), size)} images for {class_dir_path.stem}!", end="")

        print(f"\nFinished converting {class_dir_path.stem}!")


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.MODE == "all":
        apply_transformation_to_dataset(args.IN_DIRECTORY, args.OUT_DIRECTORY, "blur", args.size)
        apply_transformation_to_dataset(args.IN_DIRECTORY, args.OUT_DIRECTORY, "crop", args.size)
        apply_transformation_to_dataset(args.IN_DIRECTORY, args.OUT_DIRECTORY, "jpeg", args.size)
        apply_transformation_to_dataset(args.IN_DIRECTORY, args.OUT_DIRECTORY, "noise", args.size)
        apply_transformation_to_dataset(args.IN_DIRECTORY, args.OUT_DIRECTORY, "multi", args.size)
    else:
        if args.MODE not in {"blur", "crop", "jpeg", "noise", "multi"}:
            raise NotImplementedError("Selected unrecognized mode: {args.MODE}!")
        apply_transformation_to_dataset(args.IN_DIRECTORY, args.OUT_DIRECTORY, args.MODE, args.size, args.jpeg_output)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "MODE", help="Mode: {blur, crop, jpeg, noise, multi, all}.", type=str)
    parser.add_argument(
        "IN_DIRECTORY", help="Directory of dataset to modify.", type=str)
    parser.add_argument(
        "OUT_DIRECTORY", help="Directory to save augmented dataset.", type=str)
    parser.add_argument(
        "--size", "-s", help="Only process this amount of images.", type=int, default=None)
    parser.add_argument(
        "--seed", "-n", help="Numpy random seed", type=int, default=0)
    parser.add_argument(
        "--jpeg_output", "-j", help="Output images as .jpg instead of .png", action="store_true")
    
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
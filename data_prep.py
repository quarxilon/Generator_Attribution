"""
Adapted from original code by Frank et al., 2020:
https://github.com/RUB-SysSec/GANDCTAnalysis
"""

import argparse
import random
import functools
import numpy as np
import tensorflow as tf
import os

from pathlib import Path

from data_prep.dataset_util import image_paths, serialize_data
from data_prep.img_numpy import load_image, scale_image, dct2, normalize
from data_prep.maths import log_scale, welford


"""
Script to convert raw RGB images (post-augmentation) into numpy or tfrecord datasets.

Convention:
    Real images and binary attribution negatives use int label 0.
    Fake images and binary attribution positives use int label 1.
"""


# label convention
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1

# default dataset size per source parameters
# change via main function
TRAIN_SIZE = 7000
VAL_SIZE = 1000
TEST_SIZE = 2000


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


def _get_source_label(image_path):
    """
    Extracts the source label from an image filepath as a string.
    Output label format: ARCHITECTURE_INSTANCE
    """
    if type(image_path) is not Path:
        image_path = Path(image_path)
    source_label = image_path.parent.stem.split('_')
    if len(source_label) < 2:
        return f"{source_label[0]}"
    else:
        return f"{source_label[0]}_{source_label[1]}"


def _collect_image_paths(directory):
    """
    Collects the filepaths to every image in a specific directory (individual image sources).
    Returns a train/val/test tuple of lists of absolute Paths to the images.
    The first TRAIN_SIZE images will be used as the training set, 
    followed by the next VAL_SIZE images as the validation set,
    and then the next TEST_SIZE images as the test set.
    """
    images = list(sorted(image_paths(directory), key=_get_image_id))
    assert (len(images) >= (TRAIN_SIZE + VAL_SIZE + TEST_SIZE)), \
        f"Insufficient images in {directory}! {len(images)} < {(TRAIN_SIZE + VAL_SIZE + TEST_SIZE)}"

    train_dataset = images[:TRAIN_SIZE]
    val_dataset = images[TRAIN_SIZE: (TRAIN_SIZE + VAL_SIZE)]
    test_dataset = images[(TRAIN_SIZE + VAL_SIZE): 
                          (TRAIN_SIZE + VAL_SIZE + TEST_SIZE)]

    return (train_dataset, val_dataset, test_dataset)


def _collect_test_only_image_paths(directory):
    """
    Same as _collect_image_paths, but only handles out-of-distribution test sets.
    """
    images = list(sorted(image_paths(directory), key=_get_image_id))
    assert (len(images) >= TEST_SIZE), \
        f"Insufficient images in {directory}! {len(images)} < {TEST_SIZE}"
    test_dataset = images[:TEST_SIZE]
    return test_dataset


def _dct2_wrapper(image, log=False):
    """
    Applies the 2D discrete cosine transform on images.
    Returns the image as a numpy array.
    Also applies absolute value log-scale if log=True.
    """
    image = np.asarray(image)
    image = dct2(image)
    if log:
        image = log_scale(image)
    return image


def collect_all_paths(pos_dirs, neg_dirs, shuffle=True):
    
    """
    Collects the filepaths to every image in the positively 
    and negatively labelled directories respectively.
    
    Returns the complete training, validation, and testing sets (in order)
    as numpy arrays composed of:
        1. image filepaths (to be loaded later)
        2. their (deepfake detection) labels (0 or 1)
    
    Datasets are randomly shuffled according to the numpy random seed used.
    Set shuffle=False if shuffling is not desired.
    
    The first TRAIN_SIZE images will be used as the training set, 
    followed by the next VAL_SIZE images as the validation set,
    and then the next TEST_SIZE images as the test set.
    """
    
    pos_directories = sorted(map(str, filter(
        lambda x: x.is_dir(), Path(pos_dirs).iterdir())))
    neg_directories = sorted(map(str, filter(
        lambda x: x.is_dir(), Path(neg_dirs).iterdir())))

    train_dataset = []
    val_dataset = []
    test_dataset = []
    
    def assemble_datasets(directories, label):
        for directory in directories:
            train, val, test = _collect_image_paths(directory)
            train = zip(train, [label] * len(train))
            val = zip(val, [label] * len(val))
            test = zip(test, [label] * len(test))
            train_dataset.extend(train)
            val_dataset.extend(val)
            test_dataset.extend(test)
            del train, val, test

    assemble_datasets(pos_directories, POSITIVE_LABEL)
    assemble_datasets(neg_directories, NEGATIVE_LABEL)
        
    train_dataset = np.asarray(train_dataset)
    val_dataset = np.asarray(val_dataset)
    test_dataset = np.asarray(test_dataset)
    
    if shuffle:
        np.random.shuffle(train_dataset)
        np.random.shuffle(val_dataset)
        np.random.shuffle(test_dataset)
        
    return train_dataset, val_dataset, test_dataset


def collect_test_only_paths(test_only_dirs, test_only_positive=True, shuffle=True):
    
    """
    Same as collect_all_paths, but only handles out-of-distribution test sets.
    
    Set test_only_positive=False if the test-only sets should be negatively labelled.
    
    Datasets are randomly shuffled according to the numpy random seed used.
    Set shuffle=False if shuffling is not desired.
    """
    
    test_only_directories = sorted(map(str, filter(
        lambda x: x.is_dir(), Path(test_only_dirs).iterdir())))
    
    test_only_dataset = []
    
    for i, directory in enumerate(test_only_directories):
        test = _collect_test_only_image_paths(directory)
        if test_only_positive:
            test = zip(test, [POSITIVE_LABEL] * len(test))
        else:
            test = zip(test, [NEGATIVE_LABEL] * len(test))
        test_only_dataset.extend(test)
        del test
        
    test_only_dataset = np.asarray(test_only_dataset)
    if shuffle:
        np.random.shuffle(test_only_dataset)
    
    return test_only_dataset


def convert_images(inputs, load_function, transformation_function=None, 
                   absolute_function=None, normalize_function=None):
    
    """
    Loads an individual sample from an output array of collect_all_paths
    and applies all specified functions to the image.
    
    Returns a tuple containing the loaded and preprocessed image,
    its (deepfake detection) integer label, and (image source) string label.
    """
    
    image, label = inputs
    source_label = _get_source_label(image)
    image = load_function(image)
    
    if transformation_function is not None:
        image = transformation_function(image)
    if absolute_function is not None:
        image = absolute_function(image)
    if normalize_function is not None:
        image = normalize_function(image)

    return (image, label, source_label)


def create_directory_np(output_path, images, convert_function):
    
    """
    Applies convert_images (or equivalent functions) to an output array of collect_all_paths.
    The preprocessed dataset is then saved as NUMPY ARRAYS in the given output_path directory.
    """
    
    os.makedirs(output_path, exist_ok=True)
    print("Converting images...")
    converted_images = map(convert_function, images)

    labels = []
    src_labels = []
    
    for i, (img, label, src_label) in enumerate(converted_images):
        with open(f"{output_path}/{i}.npy", "wb+") as file:
            np.save(file, img)
            print(f"\rConverted {(i+1):6d} images!", end="")
        labels.append(label)
        src_labels.append(src_label)

    with open(f"{output_path}/labels.npy", "wb+") as file:
        np.save(file, labels)
        
    with open(f"{output_path}/source_labels.npy", "wb+") as file:
        np.save(file, src_labels)
        

def create_directory_tf(output_path, images, convert_function):
    
    """
    Applies convert_images (or equivalent functions) to an output array of collect_all_paths.
    The preprocessed dataset is serialized and saved as TFRECORDS in the given output_path directory.
    The dataset must be deserialized before training and evaluation.
    """
    
    os.makedirs(output_path, exist_ok=True)
    print("Converting images...")
    converted_images = map(convert_function, images)
    converted_images = map(serialize_data, converted_images)

    def gen():
        i = 0
        for serialized in converted_images:
            i += 1
            print(f"\rConverted {i:6d} images!", end="")
            yield serialized

    dataset = tf.data.Dataset.from_generator(
        gen, output_types=tf.string, output_shapes=())
    filename = f"{output_path}/data.tfrecords"
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset)


def normal_mode(pos_directory, neg_directory, convert_function, output_path, 
                test_only_directory=None):
    
    """
    Applies create_directory_np (which in turn applies convert_function) to all outputs of collect_all_paths.
    The preprocessed datasets are saved as NUMPY ARRAYS in subdirectories of the given output_path directory.
    """
    
    (train_dataset, val_dataset, test_dataset) = \
        collect_all_paths(pos_directory, neg_directory)
    if test_only_directory is not None:
        test_only_dataset = collect_test_only_paths(
            test_only_directory, test_only_positive=True)
    
    create_directory_np(f"{output_path}_train", train_dataset, convert_function)
    print("\nConverted training images!")
    create_directory_np(f"{output_path}_val", val_dataset, convert_function)
    print("\nConverted validation images!")
    create_directory_np(f"{output_path}_test", test_dataset, convert_function)
    print("\nConverted testing images!")
    if test_only_directory is not None:
        create_directory_np(f"{output_path}_test_only", test_only_dataset, convert_function)
        print("\nConverted testing images (out-of-distribution set)!")


def tfmode(pos_directory, neg_directory, convert_function, 
           output_path, test_only_directory=None):
    
    """
    Applies create_directory_tf (which in turn applies convert_function) to all outputs of collect_all_paths.
    The datasets are serialized and saved as TFRECORDS in subdirectories of the given output_path directory.
    The datasets must be deserialized before training and evaluation.
    """
    
    (train_dataset, val_dataset, test_dataset) = \
        collect_all_paths(pos_directory, neg_directory)
    if test_only_directory is not None:
        test_only_dataset = collect_test_only_paths(
            test_only_directory, test_only_positive=True)
    
    create_directory_tf(f"{output_path}_train_tf", train_dataset, convert_function)
    print("\nConverted training images!")
    create_directory_tf(f"{output_path}_val_tf", val_dataset, convert_function)
    print("\nConverted validation images!")
    create_directory_tf(f"{output_path}_test_tf", test_dataset, convert_function)
    print("\nConverted testing images!")
    if test_only_directory is not None:
        create_directory_tf(f"{output_path}_test_only_tf", test_only_dataset, convert_function)
        print("\nConverted testing images (out-of-distribution set)!")


    
    
def main(args):
    
    # adjust dataset size per source parameters
    global TRAIN_SIZE, VAL_SIZE, TEST_SIZE
    if args.train_size != TRAIN_SIZE:
        TRAIN_SIZE = args.train_size
    if args.val_size != VAL_SIZE:
        VAL_SIZE = args.val_size
    if args.test_size != TEST_SIZE:
        TEST_SIZE = args.test_size
    
    output = Path(args.OUT_DIRECTORY).stem
    
    load_function = functools.partial(load_image, tf=(args.mode == "tfrecords"))
    transformation_function = None
    normalize_function = None
    absolute_function = None
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # DCT coefficients or raw images?
    if args.raw:
        output += "_raw"
        
        # load RGB colour images
        if args.colour:
            load_function = functools.partial(load_function, greyscale=False)
            output += "_colour"

        # pixel input normalization scales to [-1, 1]
        if args.normalize:
            normalize_function = scale_image
            output += "_normalized"

    else: # DCT preprocessing
        output += "_dct"
        transformation_function = _dct2_wrapper

        if args.log:
            # log scale only for DCT coefficients!!!
            transformation_function = functools.partial(_dct2_wrapper, log=True)
            output += "_log_scaled"

        if args.abs:
            # scale each DCT spectrum by its max absolute value
            # based on the first half of the full training set
            train, _, _ = collect_all_paths(
                args.POS_DIRECTORY, args.NEG_DIRECTORY)
            train = train[: (len(train) * 0.5)]
            images = map(lambda x: x[0], train)
            images = map(load_function, images)
            images = map(transformation_function, images)

            first = next(images)
            current_max = np.absolute(first)
            for data in images:
                max_values = np.absolute(data)
                mask = current_max > max_values
                current_max *= mask
                current_max += max_values * ~mask

            def scale_by_absolute(image):
                return image / current_max

            absolute_function = scale_by_absolute

        if args.normalize:
            if args.log and (args.normstats != None):
                # if dctnorm statistics provided from data_statistics.py
                print("Using provided mean/var/std for dataset...")
                mean = np.load(f"{args.normstats.rstrip('/')}/mean.npy")
                std = np.load(f"{args.normstats.rstrip('/')}/std_dev.npy")
            else:
                # normalize using welford method over full training set
                train, _, _ = collect_all_paths(
                    args.POS_DIRECTORY, args.NEG_DIRECTORY)
                images = map(lambda x: x[0], train)
                images = map(load_function, images)
                images = map(transformation_function, images)
                
                if absolute_function is not None:
                    images = map(absolute_function, images)
    
                mean, var = welford(images)
                std = np.sqrt(var)
            output += "_normalized"
            normalize_function = functools.partial(
                normalize, mean=mean, std=std)

    # configures convert_images function
    convert_function = functools.partial(
        convert_images, load_function=load_function,
        transformation_function=transformation_function, 
        normalize_function=normalize_function, 
        absolute_function=absolute_function)
    
    output = f"{args.OUT_DIRECTORY.rstrip('/')}/{output}"
    
    if args.mode == "normal":
        normal_mode(args.POS_DIRECTORY, args.NEG_DIRECTORY, convert_function, 
                    output, test_only_directory=args.test_only_directory)
    elif args.mode == "tfrecords":
        tfmode(args.POS_DIRECTORY, args.NEG_DIRECTORY, convert_function, 
               output, test_only_directory=args.test_only_directory)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "POS_DIRECTORY", help="Directory of dataset (positive portion) to convert.", type=str)
    parser.add_argument(
        "NEG_DIRECTORY", help="Directory of dataset (negative portion) to convert.", type=str)
    parser.add_argument(
        "OUT_DIRECTORY", help="Directory to save converted dataset.", type=str)
    parser.add_argument(
        "--raw", "-r", help="Save image data as raw image instead of as DCT coefficients.", action="store_true")
    parser.add_argument(
        "--log", "-l", help="Log scale images (DCT coefficients only).", action="store_true")
    parser.add_argument(
        "--abs", "-a", help="Scale each feature by its max absolute value.", action="store_true")
    parser.add_argument(
        "--colour", "-c", help="Compute as RGB instead of greyscale.", action="store_true")
    parser.add_argument(
        "--normalize", "-n", help="Normalize data.", action="store_true")
    parser.add_argument(
        "--normstats", help="Directory of mean/var/std for log-norm DCT coefficients.", type=str, default=None)
    parser.add_argument(
        "--seed", "-s", help="Random seed for shuffling the dataset.", type=int, default=0)
    parser.add_argument(
        "--test_only_directory", "-t", default=None,
        help="Directory of dataset (out-of-distribution portion) to convert.", type=str)
    parser.add_argument(
        "--train_size", help="Training set size per source (class). Default: 7000", type=int, default=7000)
    parser.add_argument(
        "--val_size", help="Validation set size per source (class). Default: 1000", type=int, default=1000)
    parser.add_argument(
        "--test_size", help="Testing set size per source (class). Default: 2000", type=int, default=2000)
    

    modes = parser.add_subparsers(help="Select the mode {normal|tfrecords}", dest="mode")
    _ = modes.add_parser("normal")
    _ = modes.add_parser("tfrecords")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

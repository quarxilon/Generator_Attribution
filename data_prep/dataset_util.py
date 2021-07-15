"""
Adapted from original code by Frank et al., 2020:
https://github.com/RUB-SysSec/GANDCTAnalysis
"""

from pathlib import Path


def _find_images(data_path):
    """
    Returns list of filepaths for all images in the data_path folder.
    """
    paths = list(Path(data_path).glob("*.jpeg"))
    if len(paths) == 0:
        paths = list(Path(data_path).glob("*.jpg"))
    if len(paths) == 0:
        paths = list(Path(data_path).glob("*.png"))
    return paths


def image_paths(data_path):
    """
    Returns sorted list of absolute filepaths for all images in the data_path folder.
    """
    return [path.resolve() for path in sorted(_find_images(data_path))]


def serialize_data(data):
    """
    Serializes a single tfrecord.
    image = Bitmap of the stored image as normalized intensities.
    label = Numeric label for deepfake detection: 1=real, 0=fake.
    source = String label for image source attribution.
    shape = Tensor shape of the stored images.
    """
    import tensorflow as tf

    image, label, source = data
    feature = {
        "image": tf.train.Feature(
            float_list=tf.train.FloatList(value=image.flatten().tolist())),
        "label": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(label)])),
        "source": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[source])),
        "shape": tf.train.Feature(
            int64_list=tf.train.Int64List(value=image.shape)),
    }
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()


def deserialize_data(raw_record, shape=[256, 256, 1]):
    """
    Deserializes a single tfrecord.
    """
    import tensorflow as tf
    IMAGE_FEATURE_DESCRIPTION = {
        'image': tf.io.FixedLenFeature(shape, tf.float32),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'source': tf.io.FixedLenFeature((), tf.string),
        'shape': tf.io.FixedLenFeature((3), tf.int64),
    }

    example = tf.io.parse_single_example(raw_record, IMAGE_FEATURE_DESCRIPTION)

    image = example["image"]
    image = tf.reshape(image, shape=example["shape"])
    label = example["label"]
    source = example["source"]

    return image, label, source

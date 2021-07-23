"""
Adapted from original code by Frank et al., 2020:
https://github.com/RUB-SysSec/GANDCTAnalysis
"""

import argparse
import random

import datetime as dt
import numpy as np
import tensorflow as tf
import seaborn as sb
import matplotlib.pyplot as plt

from tqdm.keras import TqdmCallback
from pathlib import Path
from tensorflow import keras
from sklearn.metrics import confusion_matrix

from data_prep.numpy_data_gen import DataGenerator
from classifier.models import (build_detection_model,
                               build_attribution_model,
                               build_multiclass_attribution_model)
from classifier.baselines import (build_gandctanalysis_simple_cnn,
                                  build_ganfingerprints_postpool_cnn)
from utilities.confusion_matrix import make_confusion_matrix

# NOT YET IMPLEMENTED
# from data_prep.dataset_util import deserialize_data


"""
Primary script to train and test all classifier models.
"""


AUTOTUNE = tf.data.experimental.AUTOTUNE

# Global parameters
BATCH_SIZE = 256
TRAIN_SIZE = 7000
VAL_SIZE = 1000
TEST_SIZE = 2000
NUM_SOURCES = 5
INPUT_SHAPE = (256, 256, 3)
SEED = 2021
THRESHOLD = 0.5

# for FacesHQ+ dataset
SOURCE_LABEL_DICT = {
    "stylegan_tpdne"    : 1,
    "stylegan_100k"     : 2,
    "stylegan2_tpdne"   : 3
}
SOURCE_LIST = ["SG1-TPDNE", "SG1-100K", "SG2-TPDNE"]

"""
# for GAN Fingerprints dataset
SOURCE_LABEL_DICT = {
    "sngan_celeba"      : 1,
    "progan_celeba"     : 2,
    "mmdgan_celeba"     : 3,
    "cramergan_celeba"  : 4
}
SOURCE_LIST = ["SNGAN", "ProGAN", "MMDGAN", "CramerGAN"]
"""


def timestamp():
    """
    Returns current timestamp as a string.
    """
    return dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def _get_size(train=True, val=False):
    """
    Returns the total dataset size.
    
    Parameters
    ----------
    train : bool, optional
        Whether to calculate the training set size instead of the test set.
        The default is True.
    val : bool, optional
        Whether to calculate the validation set size. Takes precedence over train flag.
        The default is False.
    """
    if val:
        size = VAL_SIZE * NUM_SOURCES
    elif train:
        size = TRAIN_SIZE * NUM_SOURCES
    else:
        size = TEST_SIZE * NUM_SOURCES
    return size


def _get_num_batches_per_epoch(train=True, val=False):
    """
    Returns the number of batches per epoch: total dataset size divided by the batch size.
    
    Parameters
    ----------
    train : bool, optional
        Whether to calculate the training set size instead of the test set.
        The default is True.
    val : bool, optional
        Whether to calculate the validation set size. Takes precedence over train flag.
        The default is False.
    """
    return (_get_size(train, val) // BATCH_SIZE)


def init_tf(args):
    """
    TensorFlow configuration.
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = args.allow_growth
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    print("TensorFlow initialized!")


def load_numpy_detection(path, train=True, val=False, shuffle=True):
    
    """
    Configures and returns a minibatch loader (datagen) for the deepfake detection task.
    
    Parameters
    ----------
    path : str
        Filepath to preprocessed dataset (numpy array format) directory.
    train : bool, optional
        Whether to load the training set instead of the test set.
        The default is True.
    val : bool, optional
        Whether to load the validation set. Takes precedence over train flag.
        The default is False.
    shuffle : bool, optional
        Whether to randomly shuffle the dataset after every epoch.
        The default is True.
    """
    
    size = _get_size(train=train, val=val)
    labels = np.load(f"{path}/labels.npy")
    if len(labels) > size:
        labels = labels[:size]
    
    datagen = DataGenerator(
        path, size, labels=labels, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, 
        shuffle=shuffle, seed=SEED, attribution=False)
    print("Dataset initialized!")
    return datagen


def load_numpy_attribution(path, source=None, arch_level=False, 
                           train=True, val=False, shuffle=True, 
                           baseline_model=False, multiclass=False):
    
    """
    Configures and returns a minibatch loader (datagen) for the image source attribution task.
    
    Parameters
    ----------
    path : str
        Filepath to preprocessed dataset (numpy array format) directory.
    source : str
        Designated source of interest label for binary attribution. Not required for multiclass attribution.
    arch_level : bool, optional
        Whether to only attempt attribution at the source generator architecture level.
        The default is False.
    train : bool, optional
        Whether to load the training set instead of the test set.
        The default is True.
    val : bool, optional
        Whether to load the validation set. Takes precedence over train flag.
        The default is False.
    shuffle : bool, optional
        Whether to randomly shuffle the dataset after every epoch.
        The default is True.
    baseline_model : bool, optional
        Whether to configure the minibatch loader for baseline classifiers.
        The default is False.
    multiclass : bool, optional
        Whether to configure the minibatch loader for multiclass attribution.
        The default is False.
    """
    
    size = _get_size(train=train, val=val)
    if not (baseline_model and not multiclass):
        labels = np.load(f"{path}/labels.npy")
        if len(labels) > size:
            labels = labels[:size]
    source_labels = np.load(f"{path}/source_labels.npy")
    if len(source_labels) > size:
        source_labels = source_labels[:size]
    
    if multiclass:
        source = "MULTICLASS"
        if baseline_model:
            datagen = DataGenerator(
                path, size, labels=labels, input_shape=INPUT_SHAPE, 
                batch_size=BATCH_SIZE, shuffle=shuffle, seed=SEED, 
                attribution=True, source_labels=source_labels, 
                baseline_model=True, multiclass=True, 
                source_label_dict=SOURCE_LABEL_DICT)
        else:
            datagen = DataGenerator(
                path, size, labels=labels, input_shape=INPUT_SHAPE, 
                batch_size=BATCH_SIZE, shuffle=shuffle, seed=SEED, 
                attribution=True, source_labels=source_labels, 
                multiclass=True, source_label_dict=SOURCE_LABEL_DICT)
    elif baseline_model:
        datagen = DataGenerator(
            path, size, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, 
            shuffle=shuffle, seed=SEED, attribution=True, 
            source_labels=source_labels, chosen_source=source, 
            arch_level=arch_level, baseline_model=True)
    else:
        datagen = DataGenerator(
            path, size, labels=labels, input_shape=INPUT_SHAPE, 
            batch_size=BATCH_SIZE, shuffle=shuffle, seed=SEED, 
            attribution=True, source_labels=source_labels, 
            chosen_source=source, arch_level=arch_level)
            
    print(f"Dataset initialized! Designated source: {source}")
    return datagen


def build_model(args):
    
    """
    Builds, configures, and compiles new classifier models.
    Call this function when training models from scratch.
        
    Returns
    ----------
    model : keras.Model
        Compiled classifier model ready for training.
    model_id : str
        Model ID.
    instance_id : str
        Model instance ID.
    """
    
    # initialization
    input_shape = INPUT_SHAPE
    mirrored_strategy = tf.distribute.MirroredStrategy()
    learning_rate = args.learning_rate
    initializer = keras.initializers.RandomNormal(stddev=0.02, seed=args.seed)

    with mirrored_strategy.scope():
        
        # DEEPFAKE DETECTION MODE
        if args.MODEL_TYPE == "det":
            if args.model_class == "gdaconv":
                model = build_gandctanalysis_simple_cnn(input_shape, 1, initializer)
            elif args.model_class == "postpool":
                model = build_ganfingerprints_postpool_cnn(input_shape, 1, initializer, alpha=0.2)
            else:
                model = build_detection_model(input_shape, initializer, not args.dct)
                
        # IMAGE SOURCE ATTRIBUTION MODE
        else:
            if args.model_class == "gdaconv":
                model = build_gandctanalysis_simple_cnn(
                    input_shape, ((len(SOURCE_LABEL_DICT)+1) if args.multiclass else 1), initializer)
            elif args.model_class == "postpool":
                model = build_ganfingerprints_postpool_cnn(
                    input_shape, ((len(SOURCE_LABEL_DICT)+1) if args.multiclass else 1), initializer, alpha=0.2)
            else:
                if args.multiclass:
                    model = build_multiclass_attribution_model(
                        input_shape, num_sources=len(SOURCE_LABEL_DICT), init=initializer, gap=not(args.dct))
                else:
                    model = build_attribution_model(input_shape, init=initializer, gap=not(args.dct))
                    
                model.load_weights(args.det_model, by_name=True)
                for layer in model.layers:
                    if layer.name.startswith("d"):
                        layer.trainable = False
                        # Starting in TensorFlow 2.0, setting bn.trainable = False 
                        # will also force the batchnorm layer to run in inference mode.
                        # (Keras docs)
                    if isinstance(layer, keras.layers.BatchNormalization):
                        layer.training = False

        # Set training loss and classification metrics
        if args.model_class != "default" and args.multiclass:
            loss = keras.losses.SparseCategoricalCrossentropy()
            metric = keras.metrics.SparseCategoricalAccuracy()
        else:
            loss = keras.losses.BinaryCrossentropy()
            metric = keras.metrics.BinaryAccuracy(threshold=THRESHOLD)
            
        # binary attribution or independent multiclass attribution
        if args.model_class == "default" and args.MODEL_TYPE == "att":
            if args.multiclass:
                losses = [loss]
                for i in range(len(SOURCE_LABEL_DICT)):
                    losses.append(loss)
                model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                              loss=losses, metrics=[metric])
            else:
                model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                              loss=[loss, loss], metrics=[metric])
        # all other scenarios
        else:
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                          loss=[loss], metrics=[metric])
    
    model_id = f"{args.model_class}_{args.MODEL_TYPE}_{timestamp()}" if (args.model_id is None) else args.model_id
    instance_id = model_id if (args.instance_id is None) else args.instance_id
        
    return model, model_id, instance_id


def load_trained_model(args):
    
    """
    Load existing classifier model for retraining or evaluation.
        
    Returns
    ----------
    model : keras.Model
        Compiled classifier model ready for training.
    model_id : str
        Model ID.
    instance_id : str
        Model instance ID.
    """
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    model_path = args.model if (args.mode == "train") else args.MODEL
    with mirrored_strategy.scope():
        model = keras.models.load_model(model_path)
    if args.mode == "train": # training mode
        model_id = Path(model_path).stem if (args.model_id is None) else args.model_id
        instance_id = f"{model_id}_alt" if (args.instance_id is None) else args.instance_id
    else: # testing mode
        model_id = Path(model_path).parent.stem
        instance_id = Path(model_path).stem
    return model, model_id, instance_id


def train_model(args):
    
    """
    Training procedure.
    """
    
    # CONFIGURE MINIBATCH LOADER
    if args.MODEL_TYPE == "det": # DEEPFAKE DETECTION MODE
        train_datagen = load_numpy_detection(args.TRAIN_DATASET)
        val_datagen = load_numpy_detection(args.VAL_DATASET, val=True)
        
    elif args.model_class in ("gdaconv", "postpool"): # baseline classifier architectures
        if args.multiclass: # multiclass attribution mode
            train_datagen = load_numpy_attribution(
                args.TRAIN_DATASET, baseline_model=True, multiclass=True)
            val_datagen = load_numpy_attribution(
                args.VAL_DATASET, val=True, baseline_model=True, multiclass=True)
        else: # binary attribution mode
            train_datagen = load_numpy_attribution(args.TRAIN_DATASET, source=args.source, 
                                                   arch_level=args.arch_level, baseline_model=True)
            val_datagen = load_numpy_attribution(args.VAL_DATASET, source=args.source, 
                                                 arch_level=args.arch_level, val=True, baseline_model=True)
            
    else: # multilabel attribution using default model
        if args.multiclass:
            train_datagen = load_numpy_attribution(args.TRAIN_DATASET, multiclass=True)
            val_datagen = load_numpy_attribution(args.VAL_DATASET, val=True, multiclass=True)
        else:
            train_datagen = load_numpy_attribution(
                args.TRAIN_DATASET, source=args.source, arch_level=args.arch_level)
            val_datagen = load_numpy_attribution(
                args.VAL_DATASET, source=args.source, arch_level=args.arch_level, val=True)
    # END CONFIGURE MINIBATCH LOADER
    
    # BUILD NEW MODEL OR LOAD TRAINED MODEL
    if args.model is None:
        model, model_id, instance_id = build_model(args)
    else:
        model, model_id, instance_id = load_trained_model(args)
    
    # SET OUTPUT DIRECTORIES
    ckpt_dir = f"./checkpoints/{model_id}/{instance_id}"
    model_dir = f"./trained_models/{model_id}"
    if not Path(ckpt_dir).exists():
        Path(ckpt_dir).mkdir(parents=True)
    if not Path(model_dir).exists():
        Path(model_dir).mkdir(parents=True)
    log_path = f"./logs/{model_id}/{instance_id}"
    
    # DEFINE CALLBACKS
    if args.debug:
        callbacks = None
    else:
        callbacks = [
            keras.callbacks.TensorBoard(
                log_dir=log_path,
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', verbose=1,
                patience=args.early_stopping,
                restore_best_weights=True,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=f"{ckpt_dir}/checkpoint", verbose=1,
                save_weights_only=True, save_best_only=True, 
                monitor='val_loss', mode='min',
            ),
            TqdmCallback(verbose=0),
        ]
    
    # TRAIN AND VALIDATE MODEL
    model.summary()
    model.fit(train_datagen, epochs=args.epochs, steps_per_epoch=_get_num_batches_per_epoch(),
              verbose=0, callbacks=callbacks, validation_data=val_datagen,
              validation_freq=1, validation_steps=_get_num_batches_per_epoch(val=True))
    model_loss = model.evaluate(val_datagen, steps=_get_num_batches_per_epoch(val=True), verbose=1)[0]
    
    # SAVE TRAINED MODEL
    print(f"Saving model to {model_dir} with overall loss: {model_loss:.4f}")
    model.save(f"{model_dir}/{instance_id}.h5", save_format="h5")
    
    
def process_test_results(out_path, labels, predictions, mode=1, source="SoI", test_id=timestamp()):
    
    """
    Generates confusion matrices and classification metrics for model evaluation.
    
    Modes
    ----------
    1. Deepfake detection
    2. Binary image source attribution (Baseline models)
    3. Multiclass image source attribution (Baseline models)
    4. Binary image source attribution (Default model)
    5. Multiclass image source attribution (Default model)
    """
    
    if mode in [1, 2, 3]:
        labels = np.squeeze(labels, axis=0)
        predictions = np.squeeze(predictions, axis=0)
    else:
        secondary_labels = labels[1]
        secondary_predictions = predictions[1]
        labels = labels[0]
        predictions = predictions[0]
        
    if mode == 2:
        names = ["True negatives","False positives","False negatives","True positives"]
        categories = ["Others", source]
    elif mode == 3:
        categories = ["REAL", *SOURCE_LIST]
    else:
        names = ["True REAL","False FAKE","False REAL","True FAKE"]
        categories = ["REAL", "FAKE"]
        
    cmat = confusion_matrix(labels, predictions)
    if mode == 3:
        make_confusion_matrix(cmat, figsize=(10,7), categories=categories)
        plt.savefig(out_path.joinpath(f"ATT_{test_id}.png").__str__())
    else:
        make_confusion_matrix(cmat, group_names=names, categories=categories)
        if mode == 2:
            plt.savefig(out_path.joinpath(f"ATT_{test_id}.png").__str__())
        else:
            plt.savefig(out_path.joinpath(f"DET_{test_id}.png").__str__())
    plt.close()
    
    def compute_stupid(det_preds, att_preds):
        stupid = 0
        real_preds = 0
        for i in range(len(det_preds)):
            if det_preds[i] == 0:
                real_preds += 1
                if att_preds[i] != 0:
                    stupid += 1
        pct_stupid = stupid / real_preds
        return f"Invalid attributions: {stupid} / {pct_stupid:.2%} out of {real_preds}"
        
    
    if mode in [4, 5]:
        scmat = confusion_matrix(secondary_labels, secondary_predictions)
        if mode == 4:
            secondary_names = ["True negatives","False positives","False negatives","True positives"]
            secondary_categories = ["Others", source]
            make_confusion_matrix(scmat, group_names=secondary_names, categories=secondary_categories,
                                  extra_text=compute_stupid(predictions, secondary_predictions))
        elif mode == 5:
            secondary_categories = ["REAL", *SOURCE_LIST]
            make_confusion_matrix(scmat, figsize=(10,7), categories=secondary_categories,
                                  extra_text=compute_stupid(predictions, secondary_predictions))
        plt.savefig(out_path.joinpath(f"ATT_{test_id}.png").__str__())
        plt.close()


def test_model(args):
    
    """
    Testing procedure.
    """
    
    # CONFIGURE MINIBATCH LOADER
    if args.MODEL_TYPE == "det": # DEEPFAKE DETECTION MODE
        test_datagen = load_numpy_detection(args.TEST_DATASET, train=False, shuffle=False)
        
    elif args.model_class in ("gdaconv", "postpool"): # baseline classifier architectures
        if args.multiclass: # multiclass attribution mode
            test_datagen = load_numpy_attribution(
                args.TEST_DATASET, train=False, shuffle=False, 
                multiclass=True, baseline_model=True)
        else: # binary attribution mode
            test_datagen = load_numpy_attribution(
                args.TEST_DATASET, source=args.source, arch_level=args.arch_level, 
                train=False, shuffle=False, baseline_model=True)
            
    else: # multilabel attribution using default model
        if args.multiclass:
            test_datagen = load_numpy_attribution(
                args.TEST_DATASET, train=False, shuffle=False, multiclass=True)
        else:
            test_datagen = load_numpy_attribution(
                args.TEST_DATASET, source=args.source, arch_level=args.arch_level, 
                train=False, shuffle=False)
    # END CONFIGURE MINIBATCH LOADER

    # LOAD MODEL
    model, model_id, instance_id = load_trained_model(args)
    model.summary()
    
    # SET OUTPUT DIRECTORIES
    test_timestamp = timestamp()
    raw_results_path = Path(f"./test_results/{model_id}/{instance_id}/{test_timestamp}")
    results_path = Path(f"./test_results/{model_id}/{instance_id}")
    if not raw_results_path.exists():
        raw_results_path.mkdir(parents=True)
    # if not results_path.exists():
    #     results_path.mkdir(parents=True)
    
    # EVALUATE MODEL ON TEST SET
    raw_predictions = np.array(model.predict(test_datagen, verbose=0))
    
    # REORGANIZE MODEL OUTPUTS
    if raw_predictions.ndim < 3:
        raw_predictions = np.expand_dims(raw_predictions, axis=0) 
        """
        Initial output tensor shapes:
        --------------------
        Baselines deepfake detection and binary attribution: (1, Dataset size, 1)
        Baselines multiclass attribution: (1, Dataset size, No. of fake sources + 1)
        Default model deepfake detection: (1, Dataset size, 1)
        Default model binary attribution: (2, Dataset size, 1)
        Default model multiclass attribution: (No. of fake sources + 1, Dataset size, 1)
        """
    if args.MODEL_TYPE == "att" and args.multiclass: # Multiclass attribution
        if args.model_class == "default": # default model
            raw_predictions = np.squeeze(raw_predictions, axis=2)
            predictions = np.empty((2, raw_predictions.shape[1]), dtype=np.int8) # (2, Dataset size)
            predictions[0,:] = np.where(raw_predictions[0,:] < THRESHOLD, 0, 1).astype(np.int8)
            att_predictions = np.transpose(raw_predictions[1:,:])
            for i, pred in enumerate(att_predictions):
                pred_label = 0
                pred_high = 0.
                for j, source_pred in enumerate(pred):
                    if source_pred >= THRESHOLD and source_pred > pred_high:
                        pred_label = j + 1
                        pred_high = source_pred
                predictions[1,i] = pred_label # equivalent to argmax
        else: # baseline models
            predictions = np.argmax(raw_predictions, axis=2) # (1, Dataset size)
            raw_predictions = np.transpose(np.squeeze(raw_predictions, axis=0))
    else: # deepfake detection or binary attribution
        raw_predictions = np.squeeze(raw_predictions, axis=2)
        predictions = np.where(raw_predictions < THRESHOLD, 0, 1).astype(np.int8)
    """
    raw_predictions output shapes:
    --------------------
    Baselines deepfake detection and binary attribution: (1, Dataset size)
    Default model deepfake detection: (1, Dataset size)
    Default model binary attribution: (2, Dataset size)
    Multiclass attribution: (No. of fake sources + 1, Dataset size)
    
    predictions output shapes:
    --------------------
    Deepfake detection and binary attribution: Same as raw_predictions
    Baselines multiclass attribution: (1, Dataset size)
    Default model multiclass attribution: (2, Dataset size)
    """
    
    # PREPARE GROUND TRUTH LABELS
    labels = np.load(f"{args.TEST_DATASET}/labels.npy")
    if args.MODEL_TYPE == "att":
        source_labels = np.load(f"{args.TEST_DATASET}/source_labels.npy")
        source_of_interest = args.source
        if args.arch_level:
            source_of_interest = source_of_interest.split('_')[0]
        
    ground_truth = np.empty(predictions.shape, dtype=np.int8)
    if args.MODEL_TYPE == "att" and args.multiclass:
        if args.model_class == "default":
            for i in range(predictions.shape[1]):
                ground_truth[0,i] = labels[i]
                ground_truth[1,i] = 0 if (labels[i] == 0) else int(
                    SOURCE_LABEL_DICT[source_labels[i]])
        else:
            for i in range(predictions.shape[1]):
                ground_truth[0,i] = 0 if (labels[i] == 0) else int(
                    SOURCE_LABEL_DICT[source_labels[i]])
    else:
        for i in range(predictions.shape[1]):
            if args.MODEL_TYPE == "det" or args.model_class == "default":
                ground_truth[0,i] = labels[i]
            if args.MODEL_TYPE == "att":
                source_label = source_labels[i]
                if args.arch_level:
                    source_label = source_label.split('_')[0]
                ground_truth[(1 if (args.model_class == "default") else 0), i] = 1 if (
                    source_label == source_of_interest) else 0
    
    # PRODUCE AND SAVE TEST RESULTS GRAPHICS
    if args.MODEL_TYPE == "att":
        if args.multiclass:
            if args.model_class == "default":
                scenario_code = 5
            else:
                scenario_code = 3
        elif args.model_class == "default":
            scenario_code = 4
        else:
            scenario_code = 2
    else:
        scenario_code = 1
        
    if scenario_code in [2, 4]:
        process_test_results(results_path, ground_truth, predictions, mode=scenario_code, 
                             source=source_of_interest, test_id=test_timestamp)
    else:
        process_test_results(
            results_path, ground_truth, predictions, mode=scenario_code, test_id=test_timestamp)
    
    # SAVE RAW OUTPUTS
    print(f"Saving model predictions and ground truth labels to {raw_results_path}...")
    with open(f"{raw_results_path}/predictions.npy", "wb+") as file:
        np.save(file, predictions)
    with open(f"{raw_results_path}/raw_predictions.npy", "wb+") as file:
        np.save(file, raw_predictions)
    with open(f"{raw_results_path}/ground_truth.npy", "wb+") as file:
        np.save(file, ground_truth)


def main(args):
    
    # Input validation checks
    if args.model_class not in ("default, gdaconv, postpool"):
        raise NotImplementedError("Invalid model class specified!")
    if args.MODEL_TYPE not in ("det", "att"):
        raise NotImplementedError("Invalid model type specified!")
    if args.MODEL_TYPE == "att" and not args.multiclass and args.source is None:
        raise ValueError("No designated source label specified!")
    if args.multiclass and args.arch_level:
        raise NotImplementedError("Architecture-level multiclass attribution not yet supported!")
    init_tf(args)
    
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    if args.mode == "train": # Training mode
        if args.model is None:
            if args.MODEL_TYPE == "att" and args.model_class == "default" and args.det_model is None:
                raise ValueError("Trained detection model weights required!")
        elif not Path(args.model).exists():
            raise ValueError(f"{args.model} does not exist!")
        train_model(args)
        
    elif args.mode == "test": # Test mode
        if not Path(args.MODEL).exists():
            raise ValueError(f"{args.MODEL} does not exist!")
        test_model(args)
        
    else: # Invalid mode
        raise NotImplementedError("Invalid mode specified!")


def parse_args():
    global BATCH_SIZE, INPUT_SHAPE, NUM_SOURCES, SEED, TRAIN_SIZE, VAL_SIZE, TEST_SIZE
    
    parser = argparse.ArgumentParser()
    parser.add_argument("MODEL_TYPE",               help="Select classifier type {det, att}.", type=str)
    parser.add_argument("--model_class",    "-c",   help="Select classifier architecture {default, gdaconv, postpool}.", type=str, default="default")
    parser.add_argument("--dct",            "-f",   help="Accept DCT coefficients as input", action="store_true")
    # parser.add_argument("--tfrecords",              help="Load dataset in tfrecords format", action="store_true")
    parser.add_argument("--seed",           "-s",   help="Set random seed. Default: {SEED}", type=int, default=2021)
    parser.add_argument("--allow_growth",   "-a",   help="tf.config.gpu_options.allow_growth", action="store_true")

    commands = parser.add_subparsers(help="Mode {train|test}.", dest="mode")
    
    train = commands.add_parser("train")
    train.add_argument("TRAIN_DATASET",             help="Training dataset directory to load.", type=str)
    train.add_argument("VAL_DATASET",               help="Validation dataset directory to load.", type=str)
    train.add_argument("--model",           "-m",   help="Path to trained model HDF5 file.", type=str, default=None)
    train.add_argument("--model_id",        "-j",   help="Specify ID of model family.", type=str, default=None)
    train.add_argument("--instance_id",     "-k",   help="Specify ID of model instance/variant.", type=str, default=None)
    train.add_argument("--epochs",          "-e",   help="Epochs to train for. Default: 50.", type=int, default=50)
    train.add_argument("--image_size",      "-i",   help=f"Image size. Default: {INPUT_SHAPE[0]}", type=int, default=256)
    train.add_argument("--early_stopping",          help="Early stopping criteria. Default: 5 epochs", type=int, default=5)
    train.add_argument("--learning_rate",           help="Learning rate for Adam optimizer. Default: 0.001", type=float, default=0.001)
    train.add_argument("--num_sources",     "-n",   help=f"Number of image sources. Default: {NUM_SOURCES}", type=int, default=5)
    train.add_argument("--source",          "-l",   help="Designated source label for attribution.", type=str, default=None)
    train.add_argument("--det_model",               help="(New att models) Path to trained detection model HDF5.", type=str, default=None)
    train.add_argument("--arch_level",              help="Enable architecture level attribution. Default: instance level", action="store_true")
    train.add_argument("--greyscale",       "-g",   help="Train on greyscale images.", action="store_true")
    train.add_argument("--batch_size",      "-b",   help=f"Batch size. Default: {BATCH_SIZE}", type=int, default=256)
    train.add_argument("--multiclass",              help="Enable multiclass attribution", action="store_true")
    train.add_argument("--debug",           "-d",   help="Debug mode: no callbacks during training loop", action="store_true")
    train.add_argument("--train_size",      "-t",   help="Training set size per class. Default: 7000", type=int, default=7000)
    train.add_argument("--val_size",        "-v",   help="Validation set size per class. Default: 1000", type=int, default=1000)
    
    test = commands.add_parser("test")
    test.add_argument("MODEL",                      help="Path to trained model.", type=str)
    test.add_argument("TEST_DATASET",               help="Testing dataset to load.", type=str)
    test.add_argument("--image_size",       "-i",   help=f"Image size. Default: {INPUT_SHAPE[0]}", type=int, default=256)
    test.add_argument("--num_sources",      "-n",   help=f"Number of image sources. Default: {NUM_SOURCES}", type=int, default=5)
    test.add_argument("--source",           "-l",   help="Designated source label for attribution.", type=str)
    test.add_argument("--arch_level",               help="Enable architecture level attribution. Default: instance level", action="store_true")
    test.add_argument("--greyscale",        "-g",   help="Test on greyscale images.", action="store_true")
    test.add_argument("--batch_size",       "-b",   help=f"Batch size. Default: {BATCH_SIZE}", type=int, default=256)
    test.add_argument("--multiclass",               help="Enable multiclass attribution", action="store_true")
    test.add_argument("--test_size",        "-t",   help="Testing set size per class. Default: 2000", type=int, default=2000)

    args = parser.parse_args()
    
    BATCH_SIZE = args.batch_size
    if args.greyscale or args.dct:
        INPUT_SHAPE = (args.image_size, args.image_size, 1)
    else:
        INPUT_SHAPE = (args.image_size, args.image_size, 3)
    NUM_SOURCES = args.num_sources
    SEED = args.seed
    
    if args.mode == "train":
        TRAIN_SIZE = args.train_size
        VAL_SIZE = args.val_size
    elif args.mode == "test":
        TEST_SIZE = args.test_size

    return args


if __name__ == "__main__":
    main(parse_args())

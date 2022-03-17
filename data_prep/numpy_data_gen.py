"""
Adapted from original code in blog post by Afshine and Shervine Amidi:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

import numpy as np
import tensorflow as tf

from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    
    """
    Instantiates a Keras minibatch loader that loads dataset minibatches on the fly.
    Assumes that datasets are prepared and saved in .npy format from data_prep.py.
    Designed and documented for deepfake detection and attribution scenarios.
    """
    
    def __init__(self, dataset_path, dataset_size, labels=None, input_shape=(256,256,3), batch_size=32, 
                 shuffle=True, seed=2021, attribution=False, source_labels=None, chosen_source=None, 
                 arch_level=False, baseline=False, multiclass=False, source_id_dict=None):
        
        """
        Constructor parameters
        ----------
        dataset_path : str
            Preprocessed dataset filepath.
        dataset_size : int
            Total number of samples in dataset.
        labels : np.array (int)
            Deepfake detection labels, 0=real 1=fake. Not needed for pure attribution models.
        input_shape : (int,int,int)
            Image tensor shape. The default is (256,256,3).
        batch_size : int
            Batch size. The default is 32.
        shuffle : bool
            Whether to shuffle input images after each epoch. The default is True.
        seed : int, optional
            The default random seed is 2021.
        attribution : bool
            Whether image source attribution is being performed. The default is False.
        source_labels : np.array (str), optional
            Image source attribution labels as strings with format "architecture_instance".
            The default is None.
        chosen_source : str, optional
            Selected source of interest for binary attribution. The default is None.
        arch_level : bool, optional
            Whether to perform attribution at the source model architecture level. The default is False.
        baseline: bool, optional
            Whether a baseline classifier implementation without binary attribution capabilities is used.
            The default is False.
        multiclass : bool, optional
            Whether multiclass attribution is being performed. The default is False.
        source_id_dict : {str : int}, optional
            Dictionary mapping source_labels string values to integer IDs. The default is None.
        """
        
        self.dataset_path = dataset_path
        self.dataset_size = dataset_size
        if not (baseline and not multiclass):
            self.labels = labels
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.attribution = attribution
        np.random.seed(seed)
        self.multiclass = multiclass
        self.baseline = baseline
        
        if attribution:
            self.source_labels = source_labels
            self.arch_level = arch_level
            if multiclass:
                self.source_id_dict = source_id_dict
            elif arch_level:
                self.chosen_source = chosen_source.split('_')[0]
            else:
                self.chosen_source = chosen_source
        self.on_epoch_end()
        
        
    def __len__(self):
        """
        Returns number of batches per epoch
        """
        return int(np.floor(self.dataset_size / self.batch_size))
    
    
    def __getitem__(self, index):
        """
        Prepares one batch of data.
        Collect image indices of the current batch,
        then calls __data_generation to obtain batch with labels.
        """
        batch_indices = self.indices[
            (index * self.batch_size): ((index+1)*self.batch_size)]
        batch_data, batch_labels = self.__data_generation(batch_indices)
        return (batch_data, batch_labels)
    
    
    def on_epoch_end(self):
        """
        Updates (and shuffles) image indices after each epoch.
        """
        self.indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.shuffle:
            np.random.shuffle(self.indices)


    def __data_generation(self, batch_indices):
        
        """
        Loads one minibatch of data containing batch_size samples.
        Accepts a list of integer indices pointing to the samples.
        Returns one minibatch along with a list of label sublists.
        The first label sublist contains deepfake detection or multi-class attribution labels.
        Other label sublists are for binary attribution labels and only returned if enabled.
        """
        
        # Minibatch initialization
        batch_data = np.empty((self.batch_size, *self.input_shape))
        batch_labels = np.empty((self.batch_size))
        if self.attribution and not self.baseline:
            if self.multiclass:
                batch_source_labels = np.empty(
                    (len(self.source_id_dict), self.batch_size))
            else:
                batch_source_labels = np.empty((self.batch_size))
        
        # Generate one data sample
        for count, index in enumerate(batch_indices):
            
            # Load and store data sample
            sample = np.load(f"{self.dataset_path}/{index}.npy")
            if sample.ndim < 3:
                sample = np.expand_dims(sample, axis=2)
            batch_data[count,] = sample
            
            # Record the sample's source label as a string
            # Truncate to architectural attribution level if needed
            if self.attribution:
                source_label = self.source_labels[index]
                if self.arch_level:
                    source_label = source_label.split('_')[0]
                    
                # Process labels for multi-class attribution
                if self.multiclass:
                    if self.baseline:
                        batch_labels[count] = 0 if (self.labels[index] == 0) \
                            else self.source_id_dict[source_label]
                    else: # multiple binary attribution classifiers
                        if self.labels[index] == 0: # if real
                            batch_labels[count] = 0 # real
                            for i in range(len(self.source_id_dict)):
                                batch_source_labels[i, count] = 0
                        else:
                            batch_labels[count] = 1 # fake
                            for i in range(len(self.source_id_dict)):
                                batch_source_labels[i, count] = 1 if (
                                    (i+1) == self.source_id_dict[source_label]) else 0
                
                # Process labels for binary attribution
                elif self.baseline:
                    batch_labels[count] = 1 if (source_label == self.chosen_source) else 0
                else:
                    batch_labels[count] = self.labels[index]
                    batch_source_labels[count] = 1 if (source_label == self.chosen_source) else 0
            
            # Only process labels for deepfake detection
            else:
                batch_labels[count] = self.labels[index]
        
        # Pack minibatch tuple and return
        if self.attribution and not self.baseline:
            if self.multiclass:
                list_of_labels = [batch_labels]
                list_of_labels.extend(batch_source_labels)
                return (batch_data, list_of_labels)
            else:
                return (batch_data, [batch_labels, batch_source_labels])
        else:
            return (batch_data, [batch_labels])
        
    
    def get_output_signature(self):
        batch_data_shape = (self.batch_size, *self.input_shape)
        if self.attribution and not self.baseline:
            if self.multiclass:
                batch_labels_shape = (len(self.source_id_dict)+1, self.batch_size)
            else:
                batch_labels_shape = (2, self.batch_size)
        else:
            batch_labels_shape = (1, self.batch_size)
        return (tf.TensorSpec(shape=batch_data_shape, dtype=tf.float32), 
                tf.TensorSpec(shape=batch_labels_shape, dtype=tf.int32))
        
        
def load_id_dict(path, arch_level=False):
    """
    Imports the specified CSV file mapping sources to numeric IDs.
    """
    mappings = dict()
    csvfile = open(path)
    if arch_level:
        id_count = 0
        for line in csvfile:
            key = line.strip('\n').split(',')[0]
            key = key.split('_')[0]
            if mappings.has_key(key):
                continue
            else:
                id_count += 1
                mappings[key] = id_count
    else:
        for line in csvfile:
            (key, val) = line.strip('\n').split(',')
            mappings[key] = int(val)
    return mappings

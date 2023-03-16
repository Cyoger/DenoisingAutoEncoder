import numpy as np
import keras
import os
import tensorflow as tf
from PIL import Image

# Make this as modular as possible


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, path, batch_size=32, shuffle=True):
        self.batch_size = batch_size #how many images i want to pass
        self.list_IDs = list_IDs #corresponding list_id's of the image
        self.shuffle = shuffle
        self.path = path #
        self.on_epoch_end()

    def __len__(self):
        return int((len(self.list_IDs)/self.batch_size))

    # Load all images for the batch
    # That is, load ids self.list_IDs[index: index + batch_size]

    def __getitem__(self, index):
        #print(self.list_IDs)
        sample_id = self.list_IDs[index * self.batch_size:(index + 1) * self.batch_size]

        # load sample tifs
        self.list_IDs_temp = [k for k in sample_id]
        samples = self._generate_y(sample_id)
        noisy_samples = self._generate_X(sample_id)
        
        # All data augmentation here

        return noisy_samples, samples 

    def _generate_X(self, list_IDs_temp):
        x = []
        for i, ID in enumerate(list_IDs_temp):
            x.append(self._load_image(self.path + ID))

        x = np.array(x)
        x = self.prepro(x)
        x = self.noise(x)
        return np.array(x)
        
    def _generate_y(self, list_IDs_temp):
        y = []
        for i,ID in enumerate(list_IDs_temp):
            y.append(self._load_image(self.path + ID))
        
        y = np.array(y)
        y = self.prepro(y)
            
        return np.array(y)
        
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.list_IDs)

    def _load_image(self, path):
        with Image.open(os.path.join(path)) as img:
            
            img = img.resize((128,128))
            img_array = np.array(img)
        return img_array 
    
    def prepro(self, array):
        array = array.astype('float32')/65535.
        array = array * 2. - 1. 
        return array
    
    def noise(self, array):
        noise_factor = 0.05
        noisy_array = array + noise_factor * tf.random.normal(shape=array.shape)
        return tf.clip_by_value(noisy_array, clip_value_min=-1., clip_value_max=1.)   #better than tf.clip_by_value i think
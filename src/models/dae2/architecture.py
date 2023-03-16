import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os 
import cv2
import json

from random import shuffle
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from PIL import Image


class Network():
    def build(self):           
        input = layers.Input(shape=(128,128,1))
        #input = layers.Reshape((128,128))

        # Encoder
        x = layers.Conv2D(32, kernel_size = (3, 3), activation="tanh", padding="same")(input)
        x = layers.MaxPooling2D(pool_size =(2, 2), padding="same")(x)

        x = layers.Conv2D(64, kernel_size = (3, 3), activation="tanh", padding="same")(x)
        x = layers.MaxPooling2D(pool_size = (2, 2), padding="same")(x)


        x = layers.Conv2D(128, kernel_size = (3, 3), activation="tanh", padding="same")(x)

        # x = layers.Dense(32, activation='relu')(layers.Flatten()(x))

        # x = layers.Dense(32768, activation='relu')(x)

        # x = layers.Reshape((32, 32, 32))(x)
        # Decoder
        #x = layers.Conv2DTranspose(64, kernel_size = (3, 3), strides=2, activation="tanh", padding="same")(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(64, kernel_size = (3, 3), activation="tanh", padding="same")(x)

        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(32, kernel_size = (3, 3), activation="tanh", padding="same")(x)

        #x = layers.Conv2DTranspose(32, kernel_size = (3, 3), strides=2, activation="tanh", padding="same")(x)

        x = layers.Conv2D(1, kernel_size = (3, 3), activation="tanh", padding="same")(x)

        autoencoder = Model(input, x)
        autoencoder.compile(optimizer="adam", loss='mse')
        autoencoder.summary()
        
        return autoencoder
    
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, losses
from keras.models import Model
from random import shuffle

# load images from a directory
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        # open image file
        with Image.open(os.path.join(directory, filename)) as img:
            # convert image to numpy array
            img = img.resize((128,128))
            img_array = np.array(img)
            images.append(img_array)
    return images

# load training images
x = load_images('C:/image gen/images')
shuffle(x)
x_train = x[:int(0.8*len(x))]
x_test = x[int(0.8*len(x)):]

x_train = np.array(x_train)
x_test = np.array(x_test)

'''
# load testing images
x_test = load_images('C:/image gen/images2')
shuffle(x_test)
x_test = np.array(x_test)
'''
x_train = x_train.astype('float32') / 65535.
x_test = x_test.astype('float32') / 65535.

x_train = x_train * 2. - 1.
x_test = x_test * 2. - 1.

print (f"x_train.min(): {x_train.min()}")
print (f"x_test.min(): {x_test.min()}")

print (f"x_train.max(): {x_train.max()}")
print (f"x_test.max(): {x_test.max()}")

print (f"x_train.shape: {x_train.shape}")
print (f"x_test.shape: {x_test.shape}")

latent_dim = 64 

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(256, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(16384, activation='tanh'),
      layers.Reshape((128, 128))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())


hist = autoencoder.fit(x_train, x_train,
                epochs=100,
                shuffle=True,
                validation_data=(x_test, x_test))


encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  
  print (f"x_test[i].min(): {x_test[i].min()}")
  print (f"x_test[i].max(): {x_test[i].max()}")
  
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i] * 2 - 1)
  
  print (f"decoded_imgs[i].min(): {decoded_imgs[i].min()}")
  print (f"decoded_imgs[i].max(): {decoded_imgs[i].max()}")
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

'''
x_train = load_images('C:/image gen/images')
x_train = np.array(x_train)

# load testing images
x_test = load_images('C:/image gen/images2')
x_test = np.array(x_test)


x_train = x_train.astype('float32') / 65535.
x_test = x_test.astype('float32') / 65535.

x_train = x_train * 2. - 1.
x_test = x_test * 2. - 1.

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print(x_train.shape)
'''



noise_factor = 0.05
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape) 

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=-1., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=-1., clip_value_max=1.)




n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
plt.show()


loss_train = hist.history['loss']
loss_val = hist.history['val_loss']
epochs = range(1,101)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
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
from gen import DataGenerator


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

def noise(array):
	noise_factor = 0.05
	noisy_array = array + noise_factor * tf.random.normal(shape=array.shape) 
	return tf.clip_by_value(noisy_array, clip_value_min=-1., clip_value_max=1.)

# def rse(y_true, y_pred):
#     return K.mean(K.square(y_true - y_pred))

def rescale_images(images, original_min=0, original_max=65535):
    rescaled_images = (images + 1) * (original_max - original_min) / 2 + original_min
    rescaled_images = np.clip(rescaled_images, original_min, original_max)
    return rescaled_images






#x_train, x_test = train_test_split(x, test_size=0.1, shuffle=True)
#x_train, x_val = train_test_split(x_train, test_size=0.1, shuffle=True)

# x_train = x[:int(0.8*len(x))]
# x_test = x[int(0.8)*len(x):]


# x_train = np.array(x_train)
# x_test = np.array(x_test)

# x_train = x_train.astype('float32') / 65535.
# x_test = x_test.astype('float32') / 65535.


# x_train = x_train * 2. - 1.
# x_test = x_test * 2. - 1.




'''
print (f"x_train.min(): {x_train.min()}")
print (f"x_test.min(): {x_test.min()}")

print (f"x_train.max(): {x_train.max()}")
print (f"x_test.max(): {x_test.max()}")

print (f"x_train.shape: {x_train.shape}")
print (f"x_test.shape: {x_test.shape}")
'''


#x_train_noisy = noise(x_train)
#x_test_noisy = noise(x_test)

# x_train_noisy = np.array(x_train_noisy)
# x_test_noisy = np.array(x_test_noisy)

# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     ax = plt.subplot(1, n, i + 1)
#     plt.title("original + noise")
#     plt.imshow(tf.squeeze(x_test_noisy[i]))
#     plt.gray()
# plt.show()



input = layers.Input(shape=(128,128,1))
#input = layers.Reshape((128,128))

# Encoder
x = layers.Conv2D(32, kernel_size = (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D(pool_size =(2, 2), padding="same")(x)

x = layers.Conv2D(32, kernel_size = (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size = (2, 2), padding="same")(x)

x = layers.Dense(32, activation='relu')(layers.Flatten()(x))

x = layers.Dense(32768, activation='relu')(x)

x = layers.Reshape((32, 32, 32))(x)
# Decoder
x = layers.Conv2DTranspose(32, kernel_size = (3, 3), strides=2, activation="relu", padding="same")(x)

x = layers.Conv2DTranspose(32, kernel_size = (3, 3), strides=2, activation="relu", padding="same")(x)

x = layers.Conv2D(1, kernel_size = (3, 3), activation="tanh", padding="same")(x)

autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss='mse')
autoencoder.summary()


# train_gen = DataGenerator(
#     ["raw_image_7744.tif"],
#     "C:/Vlad Github/IDIR/Datasets/dataset_01_11_2023/images/",
# )

# sample = train_gen.__getitem__(0)

# print(sample[0][0].shape)
# print(sample[1].shape)
# cv2.imshow("lol", np.expand_dims(sample[0][0], axis=-1))
# cv2.waitKey(0)
# cv2.imshow("lol", sample[1][0])
# cv2.waitKey(0)
path = "C:/Vlad Github/IDIR/Datasets/dataset_01_11_2023/images/"
file_list = os.listdir(path)

shuffle(file_list)

training_sample_list = file_list[:int(0.8*len(file_list))]
validation_sample_list = file_list[int(0.8*len(file_list)):int(0.9*len(file_list))]
testing_sample_list = file_list[int(0.9*len(file_list)):]

json.dump(testing_sample_list, open("testing_ids.json", "w"))

training_data_gen = DataGenerator(training_sample_list, path,64)
validation_data_gen = DataGenerator(validation_sample_list, path, 64)
testing_data_gen = DataGenerator(testing_sample_list, path)
'''
x_train_noisy, x_train = data_gen.__getitem__(0) 
x_test = x_train[int(0.8*len(x)):]
x_test_noisy = x_train_noisy[int(0.8*len(x)):]

'''

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    monitor='val_loss',
    save_best_only=True,
    filepath="model_best.h5")
# x_train_noisy, x_train = train_test_split(x, test_size=0.1, shuffle=True)
# exit()
hist = autoencoder.fit(training_data_gen,
                epochs=10,
                validation_data=validation_data_gen,
                callbacks=[model_checkpoint_callback])

#save model
autoencoder.save("C:\\image gen\\src\\model.h5")

# hist = autoencoder.fit(train_data_generator, 
#                 validation_data = val_data_generator,
#                 epochs=100,
#                 batch_size=64,
#                 shuffle=True, 
#             	)


#cv2.imshow("hello", x_train[0])
# train_sample = np.expand_dims(x_train_noisy[0], axis=0)
# train_sample = np.expand_dims(train_sample, axis=-1)
# n_p = "C:\image gen\images2"
# new = load_images(n_p)

# # new = np.array(new)

# new = noise(new)
# print(new)

# prediction = autoencoder.predict(testing_data_gen)


# mean = np.mean(prediction)
# std = np.std(prediction)

# print("Mean:", mean)
# print("Standard Deviation:", std)
# #prediciton = rescale_images(prediction)
# prediction += 1
# prediction /= 2
# prediction *= 255


# dir = "C:/image gen/test_images"

# # pred_list = os.listdir()
# for i, sample in enumerate(prediction):
#     filename = f"{dir}/reconstructed_img_{i}.tif"
#     cv2.imwrite(filename ,np.squeeze(sample).astype('uint8'))

# fig, axes = plt.subplots(1, 3)

# axes[0].imshow(x_train[0])
# axes[1].imshow(np.dstack([np.squeeze(prediction[0])] * 3).astype('int32'))
# axes[2].imshow(x_train_noisy[0])

# plt.show()




# i = 0
# for i in len(x_train_noisy):
#     axes[0].imshow(x_train[i])
#     axes[1].imshow(np.squeeze(prediction))
#     axes[2].imshow(x_train_noisy[i])
#     plt.show()
#     plt.save(dir)
    	

loss_train = hist.history["loss"]
loss_val = hist.history["val_loss"]
epochs = range(1,len(loss_train) + 1)
plt.plot(epochs, loss_train, 'g', label="Training loss")
plt.plot(epochs, loss_val, 'b', label="Validation loss")
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend()
plt.show()


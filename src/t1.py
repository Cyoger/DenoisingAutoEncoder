import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import os

image_dir = "C:/image gen/images"

image_paths = []

for filename in os.listdir(image_dir):
    if filename.endswith('.tif'):
        image_paths.append(os.path.join(image_dir, filename))

def preprocess_image(image_path, new_size=(64,64)):
    # Read the image
    image = cv2.imread(image_path)
    
    # Resize the image
    image = cv2.resize(image, new_size)
    
    # Normalize the pixel values
    image = image / 255.
    return image

def preprocess_images(image_paths, new_size=(64,64)):
    preprocessed_images = list(map(lambda x: preprocess_image(x, new_size), image_paths))
    return preprocessed_images

# Preprocess images
images = preprocess_images(image_paths, new_size=(64,64))
# Define the number of classes
num_classes = 10

# Create a list to store the labels
# Create a list to store the labels
labels = []

# Iterate over the images and define the labels
for i in range(len(image_paths)):
    label = int(i % num_classes)
    labels.append(np.eye(num_classes)[label])

# Convert the labels to a numpy array
labels = np.asarray(labels)



# Split the images into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the input layer
input_img = Input(shape=(64,64,3))

# Define the encoder layers
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Define the decoder layers
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2), interpolation='nearest')(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2), interpolation='nearest')(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Create the autoencoder model
autoencoder = Model(input_img, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(X_train, y_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(X_test, y_test))
                
# Evaluate the model
score = autoencoder.evaluate(X_test, y_test, batch_size=128)

# Print the evaluation result
print("\nTest MSE:", score)

# Create the encoder model
encoder = Model(input_img, encoded)

# Create the decoder model
encoded_input = Input(shape=encoder.output_shape[1:])
decoder_layers = autoencoder.layers[-6:]
decoder = Model(encoded_input, decoder_layers(encoded_input))


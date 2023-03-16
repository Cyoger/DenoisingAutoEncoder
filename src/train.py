import importlib
import sys
import traceback
import os 
import json 
import tensorflow as tf
import matplotlib.pyplot as plt

from util import gen
from random import shuffle
from tensorflow.keras.models import Model

model_name = sys.argv[1]


try:
    Network = getattr(importlib.import_module("models." + model_name + ".architecture"), "Network")
except ModuleNotFoundError:
    traceback.print_exc()
    print("Model name does not exist.")
    exit()
            
model = Network().build()

#save to models/<model_name>/training_session/
path = "C:/Vlad Github/IDIR/Datasets/dataset_01_11_2023/images/"
file_list = os.listdir(path)

shuffle(file_list)

#train val test split
# 80    10   10
training_sample_list = file_list[:int(0.8*len(file_list))]
validation_sample_list = file_list[int(0.8*len(file_list)):int(0.9*len(file_list))]
testing_sample_list = file_list[int(0.9*len(file_list)):]

json.dump(testing_sample_list, open("C:\\image gen\\src\\models\\" + 
                                    model_name + 
                                    "\\training_session\\testing_ids.json", "w"))

training_data_gen = gen.DataGenerator(training_sample_list, path,64)
validation_data_gen = gen.DataGenerator(validation_sample_list, path, 64)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    monitor='val_loss',
    save_best_only=True,
    filepath="model_best1.h5")



hist = model.fit(training_data_gen,
                epochs=30,
                validation_data=validation_data_gen,
                callbacks=[model_checkpoint_callback])


#save model
model.save("C:\\image gen\\src\\models\\" + model_name + "\\training_session\\model.h5")

#plot loss curves 
loss_train = hist.history['loss']
loss_val = hist.history['val_loss']
epochs = range(1, len(loss_train) + 1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig("C:\\image gen\\src\\models\\" + model_name + "\\training_session\\loss_curve.png")




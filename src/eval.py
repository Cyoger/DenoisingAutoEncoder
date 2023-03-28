import tensorflow as tf
from tensorflow.keras.models import Model
import json
import numpy as np
import os
from gen import DataGenerator
import cv2


model = tf.keras.models.load_model('C:\image gen\src\model_best.h5')

model.summary()

list = json.load(open("testing_ids.json", "r"))

# print(list)

path = "C:/Vlad Github/IDIR/Datasets/dataset_01_11_2023/images/"

data_gen = DataGenerator(list, path, batch_size=1)

def change_contrast(image, contrast):    
        image = image / 255 * 2 - 1    

        image *= contrast
        
        #image = np.clip(image, -1, 1)
        
        image = np.clip((image + 1) / 2, 0, 1) * 255
        image = image.astype("uint16")
        
        return image


os.makedirs("predictions4/", exist_ok=True)
total_welds = len(list)
predictions = []


for i in range(0, len(list)):
    weld_id = int(list[i].replace("raw_image_", "").replace(".tif", ""))
        
    print("Weld", i + 1, "/", total_welds, end='\t\t\r')
    x, y = data_gen.__getitem__(i)
    
    im = np.zeros((128, 128 * 3 + 4))
    
    output = model.predict(x, verbose=0)
    output = np.squeeze(output)
    output = (output + 1) / 2 * 255
    x = (x + 1) / 2 * 255
    y = (y + 1) / 2 * 255
    
    # cv2.imshow("", output)
    
    output = change_contrast(output, 10)
    x = change_contrast(x, 10)
    y = change_contrast(y, 10)
    
    # cv2.imshow("", output.astype('uint8'))
    # cv2.waitKey(0)
    # exit()
    
    im[:, 0:128] = x
    im[:, 130: 130 + 128] = y
    
    im[:, 129 + 128 + 3:] = output
    # cv2.imshow("", im.astype('uint8'))
    # cv2.waitKey(0)
    # exit()
    #noise, target, reconstructed
    cv2.imwrite("predictions4/Weld_" + str(weld_id) + ".png", im.astype('uint8'))

# for i in range(0, len(p)):
#     p[i] = np.squeeze(p[i])

# p2 = np.squeeze(p2)


# cv2.imshow("lol", p2)
# cv2.waitKey(0)
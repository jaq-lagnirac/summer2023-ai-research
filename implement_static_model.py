# Justin Caringal
# Small-scale test of implementing a model 
# that takes picture-based inputs

import os
import sys
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf

IMG_WIDTH = 100
IMG_HEIGHT = 100
BATCH_SIZE = 5 

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

real_results = {
    'WIN_20230714_10_44_12_Pro.jpg' : 0,
    'WIN_20230714_10_44_15_Pro.jpg' : 1,
    'WIN_20230714_10_44_17_Pro.jpg' : 2,
    'WIN_20230714_10_44_20_Pro.jpg' : 3,
    'WIN_20230714_10_44_47_Pro.jpg' : 4,
    'WIN_20230714_10_44_49_Pro.jpg' : 5,
    'WIN_20230714_10_44_57_Pro.jpg' : 6,
    'WIN_20230714_10_44_58_Pro.jpg' : 7,
    'WIN_20230714_10_45_00_Pro.jpg' : 8, 
    'WIN_20230714_10_45_01_Pro.jpg' : 9,
    'IMG_1270.JPG' : 0,
    'IMG_1271.JPG' : 1,
    'IMG_1272.JPG' : 2,
    'IMG_1273.JPG' : 3,
    'IMG_1274.JPG' : 4,
    'IMG_1275.JPG' : 5,
    'IMG_1276.JPG' : 6,
    'IMG_1277.JPG' : 7,
    'IMG_1268.JPG' : 8,
    'IMG_1279.JPG' : 9,
    
}

# generate file paths
EXT = '2023-07-14-151927'
model_path = os.path.join('saved_models',
                          f'outputs_{EXT}',
                          f'saved_model_{EXT}.h5')
dir_path = os.path.join('..', 'testing_data')
test_data = os.path.join('data', 'test')

test_datagen = image.ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(
    test_data,
    target_size = (IMG_WIDTH,IMG_HEIGHT),
    batch_size = BATCH_SIZE,
    class_mode = 'categorical')

index_dict = test_generator.class_indices

print(index_dict)

# load model from location
model = load_model(model_path)

for img in os.listdir(dir_path):
    # generate file path
    img_path = os.path.join(dir_path, img)

    # load in image, convert to array
    test_image = image.load_img(img_path, target_size = (100, 100))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    # creates prediction list
    result_list = model.predict(test_image)
    # Generate arg maxes for predictions
    predicted_classes = np.argmax(result_list, axis = 1)
    print(f'Actual: {real_results[img]} --- Predicted: {predicted_classes}')
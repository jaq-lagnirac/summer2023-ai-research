# Justin Caringal
# Small-scale test of implementing a model 
# that takes picture-based inputs

import os
import sys
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf


# generate file paths
model_path = os.path.join('saved_models',
                          'outputs_2023-07-12-123433',
                          'saved_model_2023-07-12-123433.h5')
img_path = os.path.join('..', 'real_world_data', 'WIN_20230714_10_44_20_Pro.jpg')

# load model from location
model = load_model(model_path)

# load in image, convert to array
test_image = image.load_img(img_path, target_size = (100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image)
print(result) 
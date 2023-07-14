# Justin Caringal
# Small-scale test of implementing a model 
# that takes picture-based inputs

import os
import sys
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from keras.backend import set_session
import tensorflow as tf

model_path = os.path.join('saved_models',
                          'outputs_2023-07-12-123433',
                          'saved_model_2023-07-12-123433.h5')

test_img = os.path.join('..', 'real_world_data', 'WIN_20230714_10_44_20_Pro.jpg')

model = load_model(model_path)

print(model.shape)
sys.exit()

result = model.predict(test_img)

print(result)
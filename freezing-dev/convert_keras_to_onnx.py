# Justin Caringal
# Based off code from the link below
# https://github.com/Cuda-Chen/keras2onnx-example

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
import onnx
import os
os.environ['TF_KERAS'] = '1'
import keras2onnx

onnx_model_name = 'saved_model.onnx'

model_path = os.path.join('saved_models',
                          'outputs_2023-07-12-123433',
                          'saved_model_2023-07-12-123433.h5')

model = load_model(model_path)
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, onnx_model_name)
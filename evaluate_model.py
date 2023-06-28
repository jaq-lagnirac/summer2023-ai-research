# Justin Caringal
# Tests HDF5 model generated from tensorflow

import os, sys; sys
import argparse
import logging

import discord

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import typing
from tensorflow import keras as K
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

IMG_WIDTH = 100
IMG_HEIGHT = 100
BATCH_SIZE = 5 

SCRIPT_PATH = os.path.abspath(__file__)
FORMAT = '[%(asctime)s] %(levelname)s %(message)s'
l = logging.getLogger()
lh = logging.StreamHandler()
lh.setFormatter(logging.Formatter(FORMAT))
l.addHandler(lh)
l.setLevel(logging.INFO)
debug = l.debug; info = l.info; warning = l.warning; error = l.error

DESCRIPTION = '''
'''

EPILOG = '''
'''

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter):
  pass
parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
  formatter_class=CustomFormatter)

parser.add_argument('tensorflow_file')
parser.add_argument('-v', '--verbose', action='store_true',
    help='Set logging level to DEBUG')

args = parser.parse_args()

if args.verbose:
  l.setLevel(logging.DEBUG)

debug('%s begin', SCRIPT_PATH)

test_data = os.path.join('data', 'test')

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(
    test_data,
    target_size = (IMG_WIDTH,IMG_HEIGHT),
    batch_size = BATCH_SIZE,
    class_mode = 'categorical')
    
# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model(args.tensorflow_file)

# Show the model architecture
model.summary()

# Evaluate inputted model
scores = model.evaluate(test_generator)
for index, score in enumerate(scores):
  print(f'{model.metrics_names[index]}: {score}')
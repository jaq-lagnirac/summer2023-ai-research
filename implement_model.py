# Justin Caringal
# Tests HDF5 model generated from tensorflow

import argparse
import logging
import json

import cv2
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf

    

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

parser.add_argument('config',
                    help='JSON config File')
# parser.add_argument('pb',
#                     help='Tensorflow Protobuff File')
# parser.add_argument('pbtxt',
#                     help='Tensorflow Protobuff Text File')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Set logging level to DEBUG')

args = parser.parse_args()

if args.verbose:
  l.setLevel(logging.DEBUG)

with open(args.config) as fh:
  config_json = json.loads(fh.read())



debug('%s begin', SCRIPT_PATH)

checkpoint_path = "saved_models/outputs_2023-07-12-123433/checkpoints/"
checkpoint_dir = os.path.dirname(checkpoint_path)

model = models.load_model('saved_models/outputs_2023-07-12-123433/saved_model_2023-07-12-123433.h5')
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
video = cv2.VideoCapture(0)

while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into dimensions you used while training
        im = im.resize((109,82))
        img_array = np.array(im)

        #Expand dimensions to match the 4D Tensor shape.
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict function using keras
        prediction = model.predict(img_array)#[0][0]
        print(prediction)
        #Customize this part to your liking...
        if(prediction == 1 or prediction == 0):
            print("No Human")
        elif(prediction < 0.5 and prediction != 0):
            print("Female")
        elif(prediction > 0.5 and prediction != 1):
            print("Male")

        cv2.imshow("Prediction", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()

debug('%s end', SCRIPT_PATH)
# Justin Caringal
# Tests HDF5 model generated from tensorflow

import os
import sys
import argparse
import logging
import json

import numpy as np
import cv2 as cv

import typing
from tensorflow import keras as K
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras
    
    

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
parser.add_argument('--pb',
                    help='Tensorflow Protobuff File')
parser.add_argument('--pbtxt',
                    help='Tensorflow Protobuff Text File')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Set logging level to DEBUG')

args = parser.parse_args()

if args.verbose:
  l.setLevel(logging.DEBUG)

with open(args.config) as fh:
  config_json = json.loads(fh.read())

DETECTION_THRESHHOLD = 0.3
THICKNESS = 2

classes = config_json['classes']

# Colors for object labels
colors = np.random.uniform(0, 255, size=(len(classes), 3))

debug('%s begin', SCRIPT_PATH)

cam = cv.VideoCapture(0)

cvNet = cv.dnn.readNetFromTensorflow(args.pb, args.pbtxt)

while True:
    # Read in the frame
    ret_val, img = cam.read()
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

    # Run object detection
    cvOut = cvNet.forward()

    # Go through each object detected and label it
    for detection in cvOut[0, 0, : , : ]:
       score = float(detection[2])
       if score > DETECTION_THRESHHOLD:
          index = int(detection[1]) # prediction class idx
          
          # detection box bounds
          left = detection[3] * cols
          top = detection[4] * rows
          right = detection[5] * cols
          bottom = detection[6] * rows

          cv.rectangle(img,
                       (int(left), int(top)),
                       (int(right), int(bottom)),
                       (23, 230, 210),
                       thickness = THICKNESS)
          
          # draw prediction box on frame
          label = "{}: {:.2f}%".format(classes[index],
                                       score * 100)
          y = top - 15 if top - 15 > 15 else top + 15
          cv.putText(img,
                     label,
                     (int(left), int(y)),
                     cv.FONT_HERSHEY_SIMPLEX,
                     0.5,
                     colors[index], 2)

    # Display the frame
    cv.imshow('my webcam', img)

    if cv.waitKey(1) == ord('q'):
        break

# Stop filming
cam.release()
 
# Close down OpenCV
cv.destroyAllWindows()

debug('%s end', SCRIPT_PATH)
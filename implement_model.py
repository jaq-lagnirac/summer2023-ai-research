# Justin Caringal
# Tests HDF5 model generated from tensorflow

import os
import sys
import argparse
import logging

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

parser.add_argument('--pb',
                    help='Tensorflow Protobuff File')
parser.add_argument('--pbtxt',
                    help='Tensorflow Protobuff Text File')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Set logging level to DEBUG')

args = parser.parse_args()

if args.verbose:
  l.setLevel(logging.DEBUG)

debug('%s begin', SCRIPT_PATH)

cam = cv.VideoCapture(0)

tensorflowNet = cv.dnn.readNetFromTensorflow(args.pb, args.pbtxt)

while True:
    # Read in the frame
    ret_val, img = cam.read()
    rows = img.shape[0]
    cols = img.shape[1]
    tensorflowNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

    # Display the frame
    cv.imshow('my webcam', img)

    if cv.waitKey(1) == ord('q'):
        break

# Stop filming
cam.release()
 
# Close down OpenCV
cv.destroyAllWindows()

debug('%s end', SCRIPT_PATH)
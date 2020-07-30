import imutils
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet import preprocess_input
import numpy as np
import argparse
import time
import cv2
import keras
import imutils
from Fun_to_prepare_data import resize_image



def sliding_window(image, step, ws):
    # slide a window across the image
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield the current window
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])

def image_pyramid(image, scale=1.2, minSize=(224, 224)):
    # yield the original image
    yield image
    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image

def inputs(imagen):
    a=resize_image(imagen,(64,64))
    a=np.asarray(a)
    a= a.astype('float32') / 255
    return a
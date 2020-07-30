import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imutils
"""
# Read image
im = cv2.imread("../../../../Desktop/screen_naranjo.png")

# Select ROI
r = cv2.selectROI(im)
# Crop image

imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
# Display cropped image

cv2.imshow("Image", imCrop)
cv2.waitKey(100)
print(imCrop.shape)
imCrop.sort()
crop=(imCrop.shape[1],imCrop.shape[1])
print(crop)
"""
def cropi(WIDTH,image="../../../../Desktop/screen_naranjo.png"):
    im = cv2.imread(image)
    im=imutils.resize(im, width=WIDTH)
    r = cv2.selectROI(im)
    imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    imCrop.sort()
    cropi=(imCrop.shape[1],imCrop.shape[1])
    cv2.startWindowThread()
    cv2.imshow(image, imCrop)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return cropi

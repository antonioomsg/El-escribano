from __future__ import print_function
#import all the necesary libs
import time
print("importando librerias")
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pandas as pd
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import imutils
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet import preprocess_input
import numpy as np
import argparse
import keras
from Fun_to_prepare_data import * 
from helpers import *
from croppingcv2 import *


def main(args):
    print("""\ BIENVENIDO AL CONTADOR DE NARANJAS 3000    
                    ,,,.   ,@@@@@@/@@,  .oo8888o.
                    ,&%%&%&&%,@@@@@/@@@@@@,8888\88/8o
                ,%&\%&&%&&%,@@@\@@@/@@@88\88888/88'
                %&&%&%&/%&&%@@\@@/ /@@@88888\88888'
                %&&%/ %&%%&&@@\ V /@@' `88\8 `/88'
                `&%\ ` /%&'    |.|        \ '|8'
                    |o|        | |         | |
                    |.|        | |         | |
                    \\/ ._\//_/__/  ,\_//__\\/.  \_//__/_""")

    print("A continuación tendras que seleccionar con el raton una naranja, para poder saber la escala de tu imagen :)")
    wait = input("Presiona cualquier boton para continuar")


    WIDTH = 600
    PYR_SCALE = 1.5
    WIN_STEP = 5
    ROI_SIZE = cropi(WIDTH,args["image"])
    cv2.destroyAllWindows()
    INPUT_SIZE = (64, 64)

    #model = keras.models.load_model('../weights/model_full_more_oranges')
    model = keras.models.load_model('../weights/model_full_more_troncos')
    #orig = cv2.imread("../../../../Desktop/screen_naranjo.png")
    orig = cv2.imread(args["image"])
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    orig_bgr= cv2.imread(args["image"])
    #orig = Image.open("../../../../Desktop/frutilla.jpg")
    #orig = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 3)
    orig = imutils.resize(orig, width=WIDTH)
    orig_bgr=imutils.resize(orig_bgr, width=WIDTH)
    (H, W) = orig.shape[:2]

    pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)

    rois = []
    locs = []
    start = time.time()
    for image in pyramid:
        scale = W / float(image.shape[1])
        c = sliding_window(image, WIN_STEP, ROI_SIZE)
        for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
            x = int(x * scale)
            y = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)
            roi = cv2.resize(roiOrig, INPUT_SIZE)
            roi = img_to_array(roi)
            roi= roi.astype('float32') / 255
            rois.append(roi)
            locs.append((x, y, x + w, y + h))
    end = time.time()
    print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(
        end - start))

    rois = np.array(rois, dtype="float32")

    print("[INFO] classifying ROIs...")
    start = time.time()
    preds = model.predict(rois)
    end = time.time()
    print("[INFO] classifying ROIs took {:.5f} seconds".format(
        end - start))

    naranjaid="naranja_id"
    naranja_label="naranja"
    preds_map=[]

    for x,i in enumerate(preds):
        preds_map.append((naranjaid,naranja_label,preds[x][0]))

    labels = {}
    # hacemos loop sobre el modelo.
    for (i, p) in enumerate(preds_map):
        (imagenetID, label, prob) = p
        # filtramos las predicciones que son inferiores a un rango.
        if prob >= 0.99:
            # Cogemos la etiqueta de la prediccion y hallamos las coordenadas.
            box = locs[i]
            # creamos una lista con las predicciones finales, su label y las corrdenadas
            L = labels.get(label, [])
            L.append((box, prob))
            labels[label] = L

    # Hacemos un loop sobre los diferentes elementos, Naranjas, limones si hubiese...
    for label in labels.keys():
        # Hacemos una copia de la imagen y así podemos dibujar sobre ella.
        print("[INFO] Enseñando los resultado para... '{}'".format(label))
        clone = orig_bgr.copy()
        # hacemos loop de todas las imagenes para cada label.
        for (box, prob) in labels[label]:
            # dibujamos un rectangulo con las coordenadas que hemos obtenido anteriormente.
            (startX, startY, endX, endY) = box
            cv2.rectangle(clone, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
        # Enseñamos los resultados antes de realizar el maxima supresion.
        # clonamos de nuevo la imagen para poder enseñarla de nuevo mejorada.
        cv2.imshow("Before", clone)
        clone = orig_bgr.copy()

    # extraemos los rectangulo y sus predicciones y aplicamos el maxima suppresion
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, proba,overlapThresh=0.025)
    print("En total el programa ha encontrado ",len(boxes),"naranjas")
    # hacemos un loop sobre los rectangulos que nos quedaron despues de aplicar el non maxima supresion.
    for (startX, startY, endX, endY) in boxes:
        # dibujamos los rectangulos.
        cv2.rectangle(clone, (startX, startY), (endX, endY),(0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        #el codigo de abajo impreme el label para cada rectangulo, en el caso de naranjas se queda muy apretado.
        #cv2.putText(clone, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        # hacemos print despues de aplicar el maxima suppresion.

        cv2.imshow("After", clone)
    cv2.waitKey(0)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
	help="introduce el path hasta tu imagen incluida, acuerdate del formato")
    args = vars(ap.parse_args())

    main(args)


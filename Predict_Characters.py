# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 22:37:34 2021

@author: ijazt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow import keras


#Loading the model
model = keras.models.load_model('kannada_OCR.h5')

#Labels Encoding
preds = {
    0 : 'ಅ', 
    1 : 'ಆ',
    2 : 'ಇ',
    3 : 'ಈ',
    4 : 'ಉ',
    5 : 'ಊ',
    6 : 'ಋ',
    7 : 'ೠ',
    8 : 'ಎ',
    9 : 'ಏ',
    10 : 'ಐ',
    11 : 'ಒ',
    12 : 'ಓ',
    13 : 'ಔ',
    14 : 'ಅಂ',
    15 : 'ಅಃ',
    16 : 'ಕ',
    17 : 'ಖ',
    18 : 'ಗ',
    19 : 'ಘ',
    20 : 'ಙ',
    21 : 'ಚ',
    22 : 'ಛ',
    23 : 'ಜ',
    24 : 'ಝ', 
    25 : 'ಞ',
    26 : 'ಟ',
    27 : 'ಠ',
    28 : 'ಡ',
    29 : 'ಢ',
    30 : 'ಣ',
    31 : 'ತ',
    32 : 'ಥ',
    33 : 'ದ',
    34 : 'ಧ',
    35 : 'ನ',
    36 : 'ಪ',
    37 : 'ಫ',
    38 : 'ಬ',
    39 : 'ಭ',
    40 : 'ಮ',
    41 : 'ಯ',
    42 : 'ರ',
    43 : 'ಱ',
    44 : 'ಲ',
    47 : 'ವ',
    48 : 'ಶ',
    49 : 'ಷ',
    50 : 'ಸ',
    51 : 'ಹ',
    45 : 'ಳ',
    46 : 'ೞ',
    52 : '೦',
    53 : '೧',
    54 : '೨',
    55 : '೩',
    56 : '೪',
    57 : '೫',
    58 : '೬',
    59 : '೭',
    60 : '೮',
    61 : '೯'
}

img = cv2.imread('ka.jpg')

def plot_digit(img):
    plt.figure(figsize = (2, 2))
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')


def predict_character(img):
    img = cv2.resize(img, (224, 224))
    img_pred = model.predict(img.reshape(-1, 224, 224, 3))
    plot_digit(img)
    print('Predicted Character : ', preds[np.argmax(img_pred) - 1])
    
predict_character(img)
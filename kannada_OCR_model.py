import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

#Load Images

images = np.load('kannada_images.npy', allow_pickle = True) #E.g. (images, labels)
images = shuffle(shuffle(images))  #Shuffling the (images, labels) tuple




#splitting images and labels
def split_images_and_labels(imgs):
    images = [] 
    labels = []
    for img, label in imgs:
        images.append(img)
        labels.append(label)
    return images, labels

imgs, labels = split_images_and_labels(images)
imgs = np.array(imgs) / 255.0  #Normalization


#train-test-split
def train_test_split(images, labels):
    
    size = len(images)
    

    train_images = images[ : int(0.7 * size)]
    train_labels = labels[ : int(0.7 * size)]
    

    test_images = images[int(0.7 * size) : int(0.85 * size)]
    test_labels = labels[int(0.7 * size) : int(0.85 * size)]
    
    
    val_images = images[int(0.85 * size) : int(size)]
    val_labels = labels[int(0.85 * size) : int(size)]
    
    
    return train_images, train_labels, test_images,test_labels, val_images, val_labels

train_images, train_labels, test_images, test_labels, val_images, val_labels = train_test_split(imgs, labels)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)

#Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 30, 
    width_shift_range = 0.25,
    height_shift_range = 0.25, 
    horizontal_flip = True)

datagen.fit(train_images)

#ResNet-50
resnet = tf.keras.applications.ResNet50(      #Resnet50 Model
    include_top = False) 

avg = keras.layers.GlobalAveragePooling2D()(resnet.output)
output = keras.layers.Dense(63, activation = 'softmax')(avg)
model = keras.Model(inputs = resnet.inputs, outputs = output)


#Freezing the layers
for layer in resnet.layers:
    layer.trainable = False

optimizer = keras.optimizers.SGD(learning_rate = 0.2, momentum = 0.9, decay = 0.01)

model.compile(loss="sparse_categorical_crossentropy", optimizer = optimizer,
              metrics = ["accuracy"])

history = model.fit(datagen.flow(train_images, train_labels, batch_size = 32),
                    validation_data = (val_images, val_labels),
                    epochs = 5)


#Unfreezing and training the model

for layer in resnet.layers:
    layer.trainable = True

optimizer = keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9,
                                 nesterov = True, decay = 0.001)   #Nestorov Accelerated Gradient(Momentum optimization)

model.compile(loss = "sparse_categorical_crossentropy", optimizer = optimizer,
              metrics = ["accuracy"])

checkpoint = keras.callbacks.ModelCheckpoint("Kannada_OCR.h5", 
                                            save_best_only = True)

history = model.fit(datagen.flow(train_images, train_labels),
                    batch_size = 16,
                    validation_data = (val_images, val_labels),
                    epochs = 50, 
                   callbacks = [checkpoint])


resnet_model = keras.models.load_model('kannada_OCR.h5')  #loading the best saved model

resnet_model.evaluate(test_images, test_labels)  
#[0.1065862625837326, 0.9773585200309753]


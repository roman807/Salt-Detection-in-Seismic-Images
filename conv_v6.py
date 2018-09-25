# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 06:18:12 2018

@author: Roman
"""

import numpy as np
from keras.layers import Input, Conv2D#, Concatenate
from keras.models import Model
from keras import optimizers
from keras import backend as K
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import cv2
import os

tr_image_dir = 'c:\\Users\\Roman\\Documents\\Projects\\Kaggle\\TGS_Salt\\all\\train\\images'
tr_mask_dir = 'c:\\Users\\Roman\\Documents\\Projects\\Kaggle\\TGS_Salt\\all\\train\\masks'

os.chdir(tr_image_dir)
train_im = os.listdir(tr_image_dir)
x = np.array([np.array(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in train_im]) / 255

os.chdir(tr_mask_dir)
train_ma = os.listdir(tr_mask_dir)
y = np.array([np.array(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in train_ma]) / 255

# expand dimensions for CNN inout
x = np.expand_dims(x, axis=3)
y = np.expand_dims(y, axis=3)

def conv_block(x_input, num_layers, f, k):
    x = x_input
    for l in range(num_layers):
        x = Conv2D(f, (k, k), padding = 'SAME', activation = 'relu')(x)
    return x

def conv_net(input_shape, num_layers, num_filters, kernel_sizes):
    x_input = Input(input_shape)
    x = conv_block(x_input, num_layers[0], num_filters[0], kernel_sizes[0])
    x = conv_block(x, num_layers[1], num_filters[1], kernel_sizes[1])
    x = conv_block(x, num_layers[2], num_filters[2], kernel_sizes[2])
    x = conv_block(x, num_layers[3], num_filters[3], kernel_sizes[3])
    x = conv_block(x, num_layers[4], num_filters[4], kernel_sizes[4])
    x = Conv2D(1, (1,1), activation = 'sigmoid')(x)
    model = Model(inputs = x_input, outputs=x)
    return model

#input_shape = Input(shape = (101,101,1))
input_shape = (101, 101, 1)
num_layers = [1, 1, 1, 1, 1] #[5, 5, 5, 5, 5]
num_filters = [16, 8, 8, 4, 2] #[32, 24, 16, 8, 4]
kernel_sizes = [3, 5, 7, 9, 11]

#metrics IOU:
# http://www.davidtvs.com/keras-custom-metrics/
class MeanIoU(object):
    def __init__(self):
        super().__init__()  #<- super allows class for other classes to be used https://stackoverflow.com/questions/222877/what-does-super-do-in-python/33469090#33469090
    def mean_iou(self, y_true, y_pred):
        # Wraps np_mean_iou method and uses it as a TensorFlow op.
        # Takes numpy arrays as its arguments and returns numpy arrays as outputs
        return tf.py_func(self.np_mean_iou, [y_true, y_pred], tf.float32)
    def np_mean_iou(self, y_true, y_pred):
        y_pred = np.round(y_pred + 0.05, 0).reshape(-1)
        y_true = y_true.reshape(-1)        
        conf = confusion_matrix(y_pred, y_true)        
        # Compute the IoU and mean IoU from the confusion matrix:
        true_positive = conf[1,1]
        false_positive = conf[1,0]
        false_negative = conf[0,1]
        # Just in case we get a division by 0, ignore/hide the error and set the value to 0
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        iou[np.isnan(iou)] = 0
        return np.mean(iou).astype(np.float32)

miou = MeanIoU()

def bin_acc05(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred + 0.05)), axis=-1)

#model = Model(inputs = [input], outputs=[out])
model = conv_net(input_shape, num_layers, num_filters, kernel_sizes)
adam = optimizers.Adam(lr = 0.001)
model.compile(loss = 'binary_crossentropy', optimizer='adam', 
              metrics=['binary_accuracy', bin_acc05, miou.mean_iou])
model.summary()

model.fit(x, y, epochs=5, batch_size=128, validation_split=0.2, verbose=1)
y_hat = model.predict(x, verbose=1)
y_hat_binary = np.round(y_hat, 0)

# test
test_image_dir = 'c:\\Users\\Roman\\Documents\\Projects\\Kaggle\\TGS_Salt\\all\\test\\images'
os.chdir(test_image_dir)
test_im = os.listdir(test_image_dir)
x_t = np.array([np.array(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in test_im]) / 255
# expand dimensions for CNN inout
x_t = np.expand_dims(x_t, axis=3)
predict = model.predict(x_t, verbose=1)

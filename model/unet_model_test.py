#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:07:07 2018

@author: roman
"""

from keras.models import Model
from keras import optimizers, initializers
from keras import backend as K
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

initializer = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

def unet(pretrained_weights = None, input_size=(101,101,1)):
    inputs = Input(input_size)
    input_padded = ZeroPadding2D(padding=((14, 13), (14, 13)))(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=initializer)(input_padded)    
    conv11 = Conv2D(1, 1, activation='sigmoid')(conv1)
    crop = Cropping2D(cropping=((14, 13), (14, 13)))(conv11)
    model = Model(inputs=inputs, output=crop)
    return model
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:37:30 2018

@author: roman
"""

import numpy as np
import os
import cv2

def get_training_data(train_im_dir, train_ma_dir, split=0.1):
    os.chdir(train_im_dir)
    train_im = os.listdir('.')
    x1 = np.array([np.array(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in train_im]) / 255
    os.chdir('../..')
    x2 = np.flip(x1, 2)
    os.chdir(train_ma_dir)
    train_ma = os.listdir('.')
    y1 = np.array([np.array(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in train_ma]) / 255
    os.chdir('../..')
    y2 = np.flip(y1, 2) 
    x = np.append(x1, x2, axis=0)
    y = np.append(y1, y2, axis=0)
    x = np.expand_dims(x, axis=3)
    y = np.expand_dims(y, axis=3)
    x_train = np.concatenate([x[0:int(x1.shape[0]*(1-split)),:,:,:], \
            x[x1.shape[0]:int(x1.shape[0] + x1.shape[0]*(1-split)),:,:,:]])
    y_train = np.concatenate([y[0:int(y1.shape[0]*(1-split)),:,:,:], \
            y[y1.shape[0]:int(y1.shape[0] + y1.shape[0]*(1-split)),:,:,:]])
    x_val = np.concatenate([x[int(x1.shape[0]*(1-split)):int(x1.shape[0]),:,:,:], \
            x[int(x1.shape[0]+x1.shape[0]*(1-split)):,:,:,:]])
    y_val = np.concatenate([y[int(y1.shape[0]*(1-split)):int(y1.shape[0]),:,:,:], \
            y[int(y1.shape[0]+y1.shape[0]*(1-split)):,:,:,:]])
    return x_train, y_train, x_val, y_val

def get_test_data(test_image_dir):
    os.chdir(test_image_dir)
    test_im = os.listdir('.')
    x_test = np.array([np.array(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in test_im]) / 255
    os.chdir('../..')
    x_test = np.expand_dims(x_test, axis=3)
    return x_test
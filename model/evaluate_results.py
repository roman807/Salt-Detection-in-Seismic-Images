#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 06:51:01 2018

@author: roman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_mean_iou(y_train, y_train_pred, y_val, y_val_pred):
    y_train = y_train.reshape(-1)[:(400*101*101)]
    y_train_pred = y_train_pred.reshape(-1)[:(400*101*101)]
    y_val = y_val.reshape(-1)
    y_val_pred = y_val_pred.reshape(-1)
    
    true_positive_train = []
    false_positive_train = []
    false_negative_train = []
    iou_train = []
    true_positive_val = []
    false_positive_val = []
    false_negative_val = []
    iou_val = []
    thresh = [-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    for i in thresh:
        y_train_pred_bin = np.round(y_train_pred + i, 0)
        y_val_pred_bin = np.round(y_val_pred + i, 0)
        conf_train = confusion_matrix(y_train_pred_bin, y_train)
        true_positive_train.append(conf_train[1,1])
        false_positive_train.append(conf_train[1,0])
        false_negative_train.append(conf_train[0,1])
        iou_train.append(conf_train[1,1]/(conf_train[1,1]+conf_train[1,0]+conf_train[0,1]))
        conf_val = confusion_matrix(y_val_pred_bin, y_val)
        true_positive_val.append(conf_val[1,1])
        false_positive_val.append(conf_val[1,0])
        false_negative_val.append(conf_val[0,1])
        iou_val.append(conf_val[1,1]/(conf_val[1,1]+conf_val[1,0]+conf_val[0,1]))
    d = {'true_positive_train': true_positive_train, 'false_positive_train':  \
         false_positive_train, 'false_negative_train': false_negative_train, \
         'iou_train': iou_train, 'true_positive_val': true_positive_val, 'false_positive_val':  \
         false_positive_val, 'false_negative_val': false_negative_val, 'iou_val': iou_val, }
    df = pd.DataFrame.from_dict(d, orient='index')
    df.columns = ['-0.1', '-0.05', '0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35']
    plt.plot(thresh, iou_train) 
    plt.plot(thresh, iou_val)
    plt.legend(['train', 'val'])
    plt.grid()
    plt.show()
    return iou_train, iou_val

def mean_iou(threshold_adj, iou_train, iou_val):
    i = [i for i in [-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35] if i == threshold_adj][0]
    print('mean IOU training: ' + str(iou_train[i]))
    print('mean IOU validation: ' + str(iou_val[i]))

def show_pictures(x, y, y_pred):
    for i in range(len(y_pred)):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(x[i,:,:,0], cmap='binary_r', vmin=0, vmax=1)
        ax2.imshow(y[i,:,:,0], cmap='binary_r', vmin=0, vmax=1)
        ax3.imshow(y_pred[i,:,:,0], cmap='binary_r', vmin=0, vmax=1)
        plt.show()
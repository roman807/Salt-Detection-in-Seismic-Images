#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 07:47:28 2018

@author: roman
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, log_loss
import os

os.chdir('/home/roman/Downloads')
results_train = pd.read_csv('results_train.csv')
y_train = results_train['y_train']
y_train_pred = results_train['y_train_pred']
results_val = pd.read_csv('results_val.csv')
y_val = results_val['y_val']
y_val_pred = results_val['y_val_pred'].fillna(0)

true_positive_train = []
false_positive_train = []
false_negative_train = []
iou_train = []
true_positive_val = []
false_positive_val = []
false_negative_val = []
iou_val = []
thresh = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35]
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
#d = {'true_positive_val': true_positive_val, 'false_positive_val':  \
#     false_positive_val, 'false_negative_val': false_negative_val, 'iou_val': iou_val, }
df = pd.DataFrame.from_dict(d, orient='index')
df.columns = ['0.025', '0.05', '0.075', '0.1', '0.125', '0.15', '0.175', '0.2', \
              '0.225', '0.25', '0.275', '0.3', '0.325', '0.35']
plt.plot(thresh, iou_train) 
plt.plot(thresh, iou_val)
plt.legend(['train', 'val'])
plt.grid()
plt.show()

i = 0.15
y_train_pred_bin = np.round(y_train_pred + i, 0)
y_val_pred_bin = np.round(y_val_pred + i, 0)

loss_train = log_loss(y_train, y_train_pred)
loss_val = log_loss(y_val, y_val_pred)
                  
                         
print('mean IOU training: ' + str(iou_train[5]))
print('mean IOU validation: ' + str(iou_val[5]))
print('loss training: ' + str(loss_train))
print('loss validation: ' + str(loss_val))


os.chdir('/home/roman/Documents/Projects/Kaggle/TGS_Salt')
df.to_csv('Results_5b5lv2_val.csv')




















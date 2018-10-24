#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:47:27 2018

@author: roman
"""

import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import os
os.chdir('/home/roman/Documents/Projects/Kaggle/TGS_Salt/model')
import load_data
import unet_model
# import unet_model_test   # <-- use this model to test code (only 1 conv layer)
import evaluate_results
import submission

def main():
    # ---------- Load data ---------- #
    train_im_dir = './train/images/'
    train_ma_dir = './train/masks/'
    x_train, y_train, x_val, y_val = load_data.get_training_data(train_im_dir, train_ma_dir, split = 0.1)
    test_im_dir = './test/images/'
    x_test = load_data.get_test_data(test_im_dir)
    
    # ---------- Define model ---------- #
    model = unet_model.unet()
    # model = unet_model_test.unet()   # <-- use this model to test code (only 1 conv layer)
    adam = optimizers.Adam(lr = 1e-4)
    model.compile(loss = 'binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    
    # ---------- Run model ---------- #
    filepath = 'weights.best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
              mode='max')
    callbacks_list = [checkpoint]
    model.fit(x_train, y_train, epochs=1, batch_size=64, validation_data=(x_val,y_val), 
              callbacks=callbacks_list, verbose=1)
    model.load_weights('weights.best.hdf5')
    
    # ---------- Predict train results ---------- #
    y_train_pred = model.predict(x_train, verbose=1)
    y_val_pred = model.predict(x_val, verbose=1)
    
    # ---------- Evaluate results ---------- #
    y_train_eval = y_train[0:10,:,:,:]
    y_train_pred_eval = y_train_pred[0:10,:,:,:]
    iou_train, iou_val = evaluate_results.plot_mean_iou(y_train_eval, y_train_pred_eval, 
        y_val, y_val_pred)
    threshold_adj = 0
    y_train_pred = np.round(y_train_pred + threshold_adj, 0)
    y_val_pred = np.round(y_val_pred + threshold_adj, 0)
    evaluate_results.mean_iou(threshold_adj, iou_train, iou_val)
    
    # ---------- Show pictures ---------- #
    evaluate_results.show_pictures(x_val[0:10,:,:,:], y_val[0:10,:,:,:], y_val_pred[0:10,:,:,:])
    
    # ---------- Predict test results ---------- #
    y_test_pred = model.predict(x_test, verbose=1)
    y_test_pred_bin = np.round(y_test_pred + threshold_adj)
    
    # ---------- Create submission ---------- #
    file_name = 'submission.csv'
    submission.create_submission(y_test_pred_bin, test_im_dir, file_name)

if __name__ == '__main__':
	main()
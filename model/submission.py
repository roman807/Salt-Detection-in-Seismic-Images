#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 08:06:53 2018

@author: roman
"""

import numpy as np
import csv
import os

def create_submission(y_test_pred, test_dir, file_name):
    final_string = []
    for element in y_test_pred:
      locations = []
      counters = []
      loc_count = []
      array = np.transpose(element).flatten()
      if array[0]==1:
        locations.append(1)
        counters.append(1)
      for i in range(1,len(array)):
        if array[i] == 1:
          if array[i-1] == 0:
            locations.append(i+1)
            counters.append(1)
          else:
            counters[-1] += 1   
      for i in range(len(locations)):
        loc_count.append(locations[i])
        loc_count.append(counters[i])
      string = ' '.join(str(l) for l in loc_count)    
      final_string.append(string)

    os.chdir(test_dir)
    test_im = os.listdir('.')
    os.chdir('..')
    
    names = [word.split('.')[0] for word in test_im]
    with open(file_name, 'w', newline ='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(['id', 'rle_mask'])
      for i in range(len(names)):
        writer.writerow([names[i], final_string[i]])
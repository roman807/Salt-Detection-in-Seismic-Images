#!/usr/bin/env python3

"""
This script creates a (sample) CSV submission file.
Import names of the files via a command and dump in 'list.txt': 
$ ls -1 > list.txt
"""

import csv 

def format_converter(input_matrix_set):
    """
    Converts an array of matrices into an array of strings of the locations and counts of 1's: 
    i.e. the format required by kaggle.
    For e.g.
    >>> sample = np.array([[[0, 0, 1], [1, 1, 1], [0, 1, 0]],
                [[1, 0, 1], [1, 1, 1], [0, 1, 0]],
                [[1, 1, 1], [1, 0, 1], [0, 1, 0]]])
    >>> format_converter(sample)
    ['2 1 5 4', '1 2 5 4', '1 2 4 1 6 3']
    """
    final_string = []
    for element in input_matrix_set:
        
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
    
    return final_string


# Sample results
result = ['1 1']*18000

def main():
	with open('list.txt') as f:
		names = [word.split('.')[0] for word in f.readlines()]

	with open('submission.csv', 'w', newline ='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['id', 'rle_mask'])
		for i in range(len(names)):
			writer.writerow([names[i], result[i]])

if __name__ == '__main__':
	main()

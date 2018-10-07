#!/usr/bin/env python3

"""
This script creates a (sample) CSV submission file.
Import names of the files via a command and dump in 'list.txt': 
$ ls -1 > list.txt
"""

import csv 

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

import numpy as np
import csv
import time

def RSI(dataset):
	size = dataset.shape[0]-31
	binary_string = np.zeros((size,4), dtype=np.int)

	for i in range(size):
		allsum = np.zeros((4,4), dtype=np.int)

		for j in range(30): #sum the 
			index = i + 1 + j # get the index of dataset
			thisday = dataset[index]
			prvday = dataset[index+1] #previous day

			for k in range(4):
				if thisday[k+1] > prvday[k+1]:
					allsum[k][0] += thisday[k+1] - prvday[k+1]
					allsum[k][1] += 1
				else
					allsum[k][2] += prvday[k+1] - thisday[k+1]
					allsum[k][3] += 1

		for k in range(4):
			avg_up = allsum[k][0] / allsum[k][1]
			avg_down = allsum[k][2] / allsum[k][3]
			rsi = avg_up / (avg_up + avg_down) * 100
			binary_string[i][k] = rsi > 50

	return binary_string

# def priceVolumn(dataset):
# 	size = dataset.shape[0]-31
# 	binary_string = np.zeros(size, dtype=np.int)

# 	for i in range(size):
# 		index = i + 1 # get the index of dataset
# 		thisday = dataset[index]
# 		prvday = dataset[index+1] #previous day

# 		if thisday[1] > prvday[1]:
# 			if thisday[5] > prvday[5]:
# 				binary_string[i] = 0
# 			elif thisday[5] == prvday[5]:
# 				binary_string[i] = 0
# 			if thisday[5] > prvday[5]:
# 				binary_string[i] = 0




def meanAvg(dataset, M, index):
	sumClose = 0

	for i in range(M):
		sumClose += dataset[index+i][1]
	
	return sumClose / M

def probability(dataset, M):
	size = dataset.shape[0]-31
	binary_string = np.zeros(size, dtype=np.int)

	for i in range(size):
		index = i + 1 # get the index of dataset
		avg = meanAvg(dataset, M, index)
		prob = (dataset[index][1] - avg) / avg
		binary_string[i] = prob > 0

	return binary_string

def comAvg(dataset, big, small):
	size = dataset.shape[0]-31
	binary_string = np.zeros(size, dtype=np.int)

	for i in range(size):
		index = i + 1
		avg_big = meanAvg(dataset, big, index)
		avg_small = meanAvg(dataset, small, index)
		binary_string[i] = avg_small > avg_big

	return binary_string



			


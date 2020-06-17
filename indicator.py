import numpy as np
import csv
import time

def meanAvg(dataset, M, index):
	sumClose = 0

	for i in range(M):
		sumClose += dataset[index+i][1]
	
	return sumClose / M

def probability(dataset, M , pred_days = 7):
	size = dataset.shape[0]-30-pred_days
	binary_string = np.zeros(size, dtype=np.int)

	for i in range(size):
		index = i + pred_days # get the index of dataset
		avg = meanAvg(dataset, M, index)
		prob = (dataset[index][1] - avg) / avg
		binary_string[i] = prob > 0

	return binary_string

def comAvg(dataset, big, small , pred_days= 7):
	size = dataset.shape[0]-30-pred_days
	binary_string = np.zeros(size, dtype=np.int)

	for i in range(size):
		index = i + pred_days
		avg_big = meanAvg(dataset, big, index)
		avg_small = meanAvg(dataset, small, index)
		prv_big = meanAvg(dataset, big, index+1)
		prv_small = meanAvg(dataset, small, index+1)

		# if prv_small < prv_big:
		# 	binary_string[i] = avg_small > avg_big

		# else:
		# 	binary_string[i] = avg_small < avg_big

		binary_string[i] = avg_small > avg_big

	return binary_string


def RSI(dataset , pred_days = 7):
	size = dataset.shape[0]-30 - pred_days
	binary_string = np.zeros((size,4), dtype=np.int)

	for i in range(size):
		allsum = np.zeros((4,3), dtype=np.float)

		for j in range(30): #sum the 
			index = i + pred_days + j # get the index of dataset
			thisday = dataset[index]
			prvday = dataset[index+1] #previous day

			for k in range(4):
				if thisday[k+1] > prvday[k+1]:
					allsum[k][0] += thisday[k+1] - prvday[k+1]
				else:
					allsum[k][1] += prvday[k+1] - thisday[k+1]

				allsum[k][2] += 1

		for k in range(4):
			avg_up = allsum[k][0] / allsum[k][2]
			avg_down = allsum[k][1] / allsum[k][2]
			if (avg_up + avg_down) > 0:
				rsi = avg_up / (avg_up + avg_down) * 100

			else:
				rsi = 0
			binary_string[i][k] = rsi < 50

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



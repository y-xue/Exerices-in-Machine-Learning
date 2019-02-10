import random
import csv
import numpy as np

def create_dataset():
	attr_size = 2
	data_size = 1000
	train_set = [ [ 0 for j in range(attr_size) ] for i in range(data_size) ]
	for i in range(data_size):
		for j in range(attr_size):
			train_set[i][j] = random.uniform(0,30)
		if np.cov([train_set[i][0], train_set[i][1]]) > 50:
			train_set[i].append(0)
		else:
			train_set[i].append(1)

	with open("./2-train.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(train_set)

create_dataset()
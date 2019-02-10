import random
import csv

def create_dataset():
	attr_size = 50
	data_size = 1000
	train_set = [ [ 0 for j in range(attr_size) ] for i in range(data_size) ]
	for i in range(data_size):
		for j in range(attr_size):
			train_set[i][j] = random.uniform(j,j+1)
		if train_set[i][0] < 0.5:
			train_set[i].append(0)
		else:
			train_set[i].append(1)

	with open("./train.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(train_set)

create_dataset()
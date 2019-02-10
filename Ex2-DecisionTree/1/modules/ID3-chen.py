import math
from node import Node
import sys
import pickle

def handle_missing(data_set, attribute_metadata):
	if len(data_set) == 0:
		return None
	#number of attribute exclude winner
	num_attr = len(data_set[0])
	#create dictionary to handle missing value
	default = [mode(data_set)]
	for i in range(1, num_attr):
		if attribute_metadata[i]['is_nominal']:
			d = {}
			for data in data_set:
				if data[i] == None:
					continue
				elif data[i] not in d:
					d[data[i]] = 1
				else:
					d[data[i]] += 1
			max_value = 0
			max_key = 0
			for key, val in d.items():
				if val >= max_value:
					max_value = val
					max_key = key
			default.insert(i, max_key)
		else:
#			s = 0
			s = []
			for data in data_set:
				if data[i] == None:
					continue
				s.append(data[i])
#				s += data[i]
#			default.insert(i, float(s)/len(data_set))
			s.sort()
			if len(data) % 2 == 1:
				default.insert(i, s[len(data)/2])
			else:
				default.insert(i, (s[len(data)/2] + s[len(data)/2 - 1])/2)
	with open('./output/default.txt', 'wb') as f:
		pickle.dump(default, f)

	return None

def pre_handle(data_set):
	with open('./output/default.txt', 'rb') as f:
		default = pickle.load(f)
	for row in range(len(data_set)):
		for col in range(1, len(data_set[0])):
			if data_set[row][col] == None:
				data_set[row][col] = default[col]
	return data_set

def ID3_helper(data_set, attribute_metadata, numerical_splits_count, depth, default):	
	res = Node()
	if len(data_set) == 0:
		print 'length is zero!!!'
		res.label = default
	elif depth == 0:
		res.label = mode(data_set)
	elif check_homogenous(data_set) is not None:
		res.label = check_homogenous(data_set)
	else:
		best, splitting_value = pick_best_attribute(data_set, attribute_metadata, numerical_splits_count)
		if best == False:
			res.label = mode(data_set)
			return res
		res.value = mode(data_set)
		res.decision_attribute = best
		res.splitting_value = splitting_value
		res.name = attribute_metadata[best]['name']
		if attribute_metadata[best]["is_nominal"]:
			res.is_nominal = True
			dic = split_on_nominal(data_set, best)
			for key, sub_data in dic.items():
				child = ID3_helper(sub_data, attribute_metadata, numerical_splits_count, depth-1, mode(data_set))
				res.children[key] = child
		else:
			res.is_nominal = False
			tup = split_on_numerical(data_set, best, splitting_value)
			sub_data0 = tup[0]
			sub_data1 = tup[1]
			numerical_splits_count[best] -= 1
			child0 = ID3_helper(sub_data0, attribute_metadata, numerical_splits_count, depth-1, mode(data_set))
			child1 = ID3_helper(sub_data1, attribute_metadata, numerical_splits_count, depth-1, mode(data_set))
			res.children[0] = child0
			res.children[1] = child1
	return res
	
def ID3(data_set, attribute_metadata, numerical_splits_count, depth):
    # Your code here
    print '*'
    # handle_missing(data_set, attribute_metadata)
    # data_set = pre_handle(data_set)
    return ID3_helper(data_set, attribute_metadata, numerical_splits_count, depth, 0)

def check_homogenous(data_set):
    # Your code here
	if len(data_set) == 1:
		return data_set[0][0]
	val = data_set[0][0]
	for data in data_set:
		if data[0] != val:
			return None
	return val
# ======== Test Cases =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  None
# data_set = [[0],[1],[None],[0]]
# check_homogenous(data_set) ==  None
# data_set = [[1],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  1

def pick_best_attribute(data_set, attribute_metadata, numerical_splits_count):
    # Your code here
	step = int(len(data_set) * 0.01)
	if step == 0:
		step = 1
	index = False
	splitValue = False
	maxGain = 0
	for i in range(1, len(attribute_metadata)):
		if attribute_metadata[i]['is_nominal']:
			gain = gain_ratio_nominal(data_set, i)
			if gain > maxGain:
				maxGain = gain
				index = i
				splitValue = False
		elif numerical_splits_count[i] > 0:
			gain, threshold  = gain_ratio_numeric(data_set, i, step)
			if gain > maxGain:
				maxGain = gain
				index = i
				splitValue = threshold
	return index, splitValue

# # ======== Test Cases =============================
# numerical_splits_count = [20,20]
# attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
# data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [0, 0.51], [1, 0.4]]
# pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, 0.51)
# attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "weather",'is_nominal': True}]
# data_set = [[0, 0], [1, 0], [0, 2], [0, 2], [0, 3], [1, 1], [0, 4], [0, 2], [1, 2], [1, 5]]
# pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, False)

# Uses gain_ratio_nominal or gain_ratio_numeric to calculate gain ratio.

def mode(data_set):
    # Your code here
	num_pos = 0
	num_neg = 0
	for data in data_set:
		if data[0] == 1:
			num_pos = num_pos + 1
		elif data[0] == 0:
			num_neg = num_neg + 1
	if num_pos > num_neg:
		return 1
	else:
		return 0
# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# mode(data_set) == 1
# data_set = [[0],[1],[0],[0]]
# mode(data_set) == 0

def entropy(data_set):
	#Your code here
	num_pos = 0
	num_neg = 0
	for data in data_set:
		if data[0] == 1:
			num_pos = num_pos + 1
		elif data[0] == 0:
			num_neg = num_neg + 1

	p_pos = num_pos / float(num_pos + num_neg)
	p_neg = num_neg / float(num_pos + num_neg)
	if p_pos == 0 or p_neg == 0:
		return 0
	entropy = -p_pos * math.log(p_pos, 2) - p_neg * math.log(p_neg, 2)
	return entropy
# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[0],[1],[1],[1]]
# entropy(data_set) == 0.811
# data_set = [[0],[0],[1],[1],[0],[1],[1],[0]]
# entropy(data_set) == 1.0
# data_set = [[0],[0],[0],[0],[0],[0],[0],[0]]
# entropy(data_set) == 0


def gain_ratio_nominal(data_set, attribute):
    # Your code here
	d = split_on_nominal(data_set, attribute)
	splitEntropy = 0
	Entropy = 0
	for key in d:
		Sv = float(len(d[key]))
		S = float(len(data_set))
		splitEntropy = splitEntropy - (Sv/S)*math.log(Sv/S, 2)
		Entropy = Entropy + (Sv/S)*entropy(d[key])
	gain = entropy(data_set) - Entropy
	if splitEntropy == 0:
		return 0
	return gain / splitEntropy
# ======== Test case =============================
# data_set, attr = [[1, 2], [1, 0], [1, 0], [0, 2], [0, 2], [0, 0], [1, 3], [0, 4], [0, 3], [1, 1]], 1
# gain_ratio_nominal(data_set,attr) == 0.11470666361703151
# data_set, attr = [[1, 2], [1, 2], [0, 4], [0, 0], [0, 1], [0, 3], [0, 0], [0, 0], [0, 4], [0, 2]], 1
# gain_ratio_nominal(data_set,attr) == 0.2056423328155741
# data_set, attr = [[0, 3], [0, 3], [0, 3], [0, 4], [0, 4], [0, 4], [0, 0], [0, 2], [1, 4], [0, 4]], 1
# gain_ratio_nominal(data_set,attr) == 0.06409559743967516

def gain_ratio_numeric(data_set, attribute, steps):
    # Your code here
	finalThreshold = 0
	finalGainRatio = 0
	for i in range(0, len(data_set), steps):
		threshold = data_set[i][attribute]
		d = split_on_numerical(data_set, attribute, threshold)
		if len(d[0]) == 0 or len(d[1]) == 0:
			continue
		S = float(len(data_set))
		S0 = float(len(d[0]))
		S1 = float(len(d[1]))
		splitEntropy = -S0/S*math.log(S0/S, 2) - S1/S*math.log(S1/S, 2)
		gain = entropy(data_set) - S0/S*entropy(d[0]) - S1/S*entropy(d[1])
		ratio = gain/splitEntropy
		if ratio > finalGainRatio:
			finalGainRatio = ratio
			finalThreshold = threshold

	return finalGainRatio, finalThreshold

# ======== Test case =============================
# data_set,attr,step = [[1,0.05], [1,0.17], [1,0.64], [0,0.38], [0,0.19], [1,0.68], [1,0.69], [1,0.17], [1,0.4], [0,0.53]], 1, 2
# gain_ratio_numeric(data_set,attr,step) == (0.21744375685031775, 0.64)
# data_set,attr,step = [[1, 0.35], [1, 0.24], [0, 0.67], [0, 0.36], [1, 0.94], [1, 0.4], [1, 0.15], [0, 0.1], [1, 0.61], [1, 0.17]], 1, 4
# gain_ratio_numeric(data_set,attr,step) == (0.11689800358692547, 0.94)
# data_set,attr,step = [[1, 0.1], [0, 0.29], [1, 0.03], [0, 0.47], [1, 0.25], [1, 0.12], [1, 0.67], [1, 0.73], [1, 0.85], [1, 0.25]], 1, 1
# gain_ratio_numeric(data_set,attr,step) == (0.23645279766002802, 0.29)

def split_on_nominal(data_set, attribute):
    # Your code here
	d = {}
	for data in data_set:
		if not data[attribute] in d:
			d[data[attribute]] = [data]
		else:
			d[data[attribute]].append(data)
	return d
# ======== Test case =============================
# data_set, attr = [[0, 4], [1, 3], [1, 2], [0, 0], [0, 0], [0, 4], [1, 4], [0, 2], [1, 2], [0, 1]], 1
# split_on_nominal(data_set, attr) == {0: [[0, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3]], 4: [[0, 4], [0, 4], [1, 4]]}
# data_set, attr = [[1, 2], [1, 0], [0, 0], [1, 3], [0, 2], [0, 3], [0, 4], [0, 4], [1, 2], [0, 1]], 1
# split on_nominal(data_set, attr) == {0: [[1, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3], [0, 3]], 4: [[0, 4], [0, 4]]}

def split_on_numerical(data_set, attribute, splitting_value):
    # Your code here
	d = ([], [])
	for data in data_set:
		if data[attribute] < splitting_value:
			d[0].append(data)
		else:
			d[1].append(data)
	return d
# ======== Test case =============================
# d_set,a,sval = [[1, 0.25], [1, 0.89], [0, 0.93], [0, 0.48], [1, 0.19], [1, 0.49], [0, 0.6], [0, 0.6], [1, 0.34], [1, 0.19]],1,0.48
# split_on_numerical(d_set,a,sval) == ([[1, 0.25], [1, 0.19], [1, 0.34], [1, 0.19]],[[1, 0.89], [0, 0.93], [0, 0.48], [1, 0.49], [0, 0.6], [0, 0.6]])
# d_set,a,sval = [[0, 0.91], [0, 0.84], [1, 0.82], [1, 0.07], [0, 0.82],[0, 0.59], [0, 0.87], [0, 0.17], [1, 0.05], [1, 0.76]],1,0.17
# split_on_numerical(d_set,a,sval) == ([[1, 0.07], [1, 0.05]],[[0, 0.91],[0, 0.84], [1, 0.82], [0, 0.82], [0, 0.59], [0, 0.87], [0, 0.17], [1, 0.76]])

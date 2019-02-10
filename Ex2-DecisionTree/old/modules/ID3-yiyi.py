import math
from node import Node
import sys
from sets import Set


def ID3(data_set, attribute_metadata, numerical_splits_count, depth):
    '''
    See Textbook for algorithm.
    Make sure to handle unknown values, some suggested approaches were
    given in lecture.
    ========================================================================================================
    Input:  A data_set, attribute_metadata, maximum number of splits to consider for numerical attributes,
	maximum depth to search to (depth = 0 indicates that this node should output a label)
    ========================================================================================================
    Output: The node representing the decision tree learned over the given data set
    ========================================================================================================

    '''
    # Your code here
    root = Node()
    if len(data_set) == 0:
        return root
    else :
        root.label = check_homogenous(data_set)
        if root.label != None:
            return root
        else :
            if depth == 0 :
                root.label = mode(data_set)
                return root
            else :
                best = pick_best_attribute(data_set, attribute_metadata, numerical_splits_count)
                if best[0] == False :   # include attributes is empty
                    root.label = mode(data_set)
                elif best[1] == False : # nominal
                    split = split_on_nominal(data_set, best[0])
                    numerical_splits_count[best[0]] = 0
                    for x in split.keys():
                        root.children[x] = ID3(split[x],attribute_metadata, numerical_splits_count, depth-1)
                    root.decision_attribute = best[0]
                    root.is_nominal = True
                    root.name = attribute_metadata[best[0]]['name']
                    root.value = mode(data_set)
                    root.splitting_value = False
                else :
                    split = split_on_numerical(data_set, best[0], best[1])
                    numerical_splits_count[best[0]] -= 1
                    root.children[0] = ID3(split[0],attribute_metadata, numerical_splits_count, depth-1)
                    root.children[1] = ID3(split[1],attribute_metadata, numerical_splits_count, depth-1)
                    root.decision_attribute = best[0]
                    root.is_nominal = False
                    root.splitting_value = best[1]
                    root.name = attribute_metadata[best[0]]['name']
                    root.value = mode(data_set)
            return root



def check_homogenous(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Checks if the output value (index 0) is the same for all examples in the the data_set, if so return 
    that output value, otherwise return None.
    ========================================================================================================
    Output: Return either the homogenous attribute or None
    ========================================================================================================
     '''
    # Your code here
    if len(data_set) == 0:
        return None
    homo = data_set[0][0]
    for item in data_set:
        if item[0] != homo:
            return None
    return homo
# ======== Test Cases =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  None
# data_set = [[0],[1],[None],[0]]
# check_homogenous(data_set) ==  None
# data_set = [[1],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  1

def pick_best_attribute(data_set, attribute_metadata, numerical_splits_count):
    '''
    ========================================================================================================
    Input:  A data_set, attribute_metadata, splits counts for numeric
    ========================================================================================================
    Job:    Find the attribute that maximizes the gain ratio. If attribute is numeric return best split value.
            If nominal, then split value is False.
            If gain ratio of all the attributes is 0, then return False, False
            Only consider numeric splits for which numerical_splits_count is greater than zero
    ========================================================================================================
    Output: best attribute, split value if numeric
    ========================================================================================================
    '''
    # Your code here
    al = len(attribute_metadata)
    best_ratio = [0,0]
    ratio = []
    iattr = 0
    for i in xrange(1, al):
        if numerical_splits_count[i] > 0:
            if attribute_metadata[i]['is_nominal']:
                ratio = [gain_ratio_nominal(data_set, i), False]
            else :
                ratio = gain_ratio_numeric(data_set, i, 403)
            if best_ratio[0] < ratio[0]:
                best_ratio = ratio
                iattr = i
    #numerical_splits_count[iattr] -= 1
    if best_ratio[0] == 0:
        return (False, False)
    return (iattr,best_ratio[1])


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
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Takes a data_set and finds mode of index 0.
    ========================================================================================================
    Output: mode of index 0.
    ========================================================================================================
    '''
    # Your code here
    # majority element moore voting
    common = data_set[0][0]
    count = 1
    datalength = len(data_set)
    for i in xrange(1, datalength): 
        if(data_set[i][0] == common):
            count += 1;
        else:
            count -= 1;
        if(count == 0):
            common = data_set[i][0]
            count = 1;
    return common

# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# mode(data_set) == 1
# data_set = [[0],[1],[0],[0]]
# mode(data_set) == 0

def entropy(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Calculates the entropy of the attribute at the 0th index, the value we want to predict.
    ========================================================================================================
    Output: Returns entropy. See Textbook for formula
    ========================================================================================================
    '''
    datalength = len(data_set)
    hash_table = {}
    for i in xrange(0, datalength):
        if hash_table.has_key(data_set[i][0]):
            hash_table[data_set[i][0]] += 1.0
        else :
            hash_table[data_set[i][0]] = 1.0
    res = 0.0
    for i in hash_table.keys():
        temp = hash_table[i] / datalength
        res = res - temp * math.log(temp) / math.log(2)
    return res


# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[0],[1],[1],[1]]
# entropy(data_set) == 0.811
# data_set = [[0],[0],[1],[1],[0],[1],[1],[0]]
# entropy(data_set) == 1.0
# data_set = [[0],[0],[0],[0],[0],[0],[0],[0]]
# entropy(data_set) == 0


def gain_ratio_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  Subset of data_set, index for a nominal attribute
    ========================================================================================================
    Job:    Finds the gain ratio of a nominal attribute in relation to the variable we are training on.
    ========================================================================================================
    Output: Returns gain_ratio. See https://en.wikipedia.org/wiki/Information_gain_ratio
    ========================================================================================================
    '''
    # Your code here
    dic = {}
    datalength = len(data_set)
    for i in xrange(datalength):
        if dic.has_key(data_set[i][attribute]):
            dic[data_set[i][attribute]].append([data_set[i][0]])
        else :
            dic[data_set[i][attribute]] = [[data_set[i][0]]]
    iv = entropy([[x[attribute]] for x in data_set])

    ig_before = entropy([[y[0]] for y in data_set])
    ig_after = 0.0
    for i in dic.keys():
        ig_after += entropy(dic[i]) * len(dic[i]) / datalength
    if ig_before - ig_after == 0.0 or iv == 0:
        return 0
    else:
        return (ig_before - ig_after) / iv

# ======== Test case =============================
# data_set, attr = [[1, 2], [1, 0], [1, 0], [0, 2], [0, 2], [0, 0], [1, 3], [0, 4], [0, 3], [1, 1]], 1
# gain_ratio_nominal(data_set,attr) == 0.11470666361703151
# data_set, attr = [[1, 2], [1, 2], [0, 4], [0, 0], [0, 1], [0, 3], [0, 0], [0, 0], [0, 4], [0, 2]], 1
# gain_ratio_nominal(data_set,attr) == 0.2056423328155741
# data_set, attr = [[0, 3], [0, 3], [0, 3], [0, 4], [0, 4], [0, 4], [0, 0], [0, 2], [1, 4], [0, 4]], 1
# gain_ratio_nominal(data_set,attr) == 0.06409559743967516

def gain_ratio_numeric(data_set, attribute, steps):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, and a step size for normalizing the data.
    ========================================================================================================
    Job:    Calculate the gain_ratio_numeric and find the best single threshold value
            The threshold will be used to split examples into two sets
                 those with attribute value GREATER THAN OR EQUAL TO threshold
                 those with attribute value LESS THAN threshold
            Use the equation here: https://en.wikipedia.org/wiki/Information_gain_ratio
            And restrict your search for possible thresholds to examples with array index mod(step) == 0
    ========================================================================================================
    Output: This function returns the gain ratio and threshold value
    ========================================================================================================
    '''
    # Your code here
    datalength = len(data_set)
    data_hash = set()
    H_before = entropy(data_set)
    for i in xrange(0, datalength):
        if i % steps == 0 :
            data_hash.add(data_set[i][attribute])
    max_ = 0
    sval = 0
    for j in data_hash:
        data0 = []
        data1 = []
        for i in xrange(0, datalength):
            if  data_set[i][attribute] < j : 
                data0.append([data_set[i][0]])
            else :
                data1.append([data_set[i][0]])
        p0 = (len(data0)/float(datalength))
        p1 = (len(data1)/float(datalength))
        if len(data1) != datalength :
            H_after = p0 * entropy(data0) + p1 * entropy(data1)
            ig = H_before - H_after
            iv = 0
            if p0 != 0 :
                iv += -(p0 * math.log(p0) / math.log(2))
            if p1 != 0 :
                iv += -(p1 * math.log(p1) / math.log(2))
            IGR = ig / iv
            if max_ < IGR :
                max_ = IGR
                sval = j
    #print [max_, threshold_]
    return [max_, sval]

# ======== Test case =============================
# data_set,attr,step = [[0,0.05], [1,0.17], [1,0.64], [0,0.38], [0,0.19], [1,0.68], [1,0.69], [1,0.17], [1,0.4], [0,0.53]], 1, 2
# gain_ratio_numeric(data_set,attr,step) == (0.31918053332474033, 0.64)
# data_set,attr,step = [[1, 0.35], [1, 0.24], [0, 0.67], [0, 0.36], [1, 0.94], [1, 0.4], [1, 0.15], [0, 0.1], [1, 0.61], [1, 0.17]], 1, 4
# gain_ratio_numeric(data_set,attr,step) == (0.11689800358692547, 0.94)
# data_set,attr,step = [[1, 0.1], [0, 0.29], [1, 0.03], [0, 0.47], [1, 0.25], [1, 0.12], [1, 0.67], [1, 0.73], [1, 0.85], [1, 0.25]], 1, 1
# gain_ratio_numeric(data_set,attr,step) == (0.23645279766002802, 0.29)

def split_on_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  subset of data set, the index for a nominal attribute.
    ========================================================================================================
    Job:    Creates a dictionary of all values of the attribute.
    ========================================================================================================
    Output: Dictionary of all values pointing to a list of all the data with that attribute
    ========================================================================================================
    '''
    # Your code here
    dic = {}
    datalength = len(data_set)
    for i in xrange(0, datalength):
        if dic.has_key(data_set[i][attribute]):
            dic[data_set[i][attribute]].append(data_set[i])
        else:
            dic[data_set[i][attribute]] = [data_set[i]]
    return dic

# ======== Test case =============================
# data_set, attr = [[0, 4], [1, 3], [1, 2], [0, 0], [0, 0], [0, 4], [1, 4], [0, 2], [1, 2], [0, 1]], 1
# split_on_nominal(data_set, attr) == {0: [[0, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3]], 4: [[0, 4], [0, 4], [1, 4]]}
# data_set, attr = [[1, 2], [1, 0], [0, 0], [1, 3], [0, 2], [0, 3], [0, 4], [0, 4], [1, 2], [0, 1]], 1
# split on_nominal(data_set, attr) == {0: [[1, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3], [0, 3]], 4: [[0, 4], [0, 4]]}

def split_on_numerical(data_set, attribute, splitting_value):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, threshold (splitting) value
    ========================================================================================================
    Job:    Splits data_set into a tuple of two lists, the first list contains the examples where the given
	attribute has value less than the splitting value, the second list contains the other examples
    ========================================================================================================
    Output: Tuple of two lists as described above
    ========================================================================================================
    '''
    # Your code here
    dic = ([],[])
    datalength = len(data_set)
    for i in xrange(0, datalength):
        if data_set[i][attribute] < splitting_value:
            dic[0].append(data_set[i])
        else:
            dic[1].append(data_set[i])

    return dic
# ======== Test case =============================
# d_set,a,sval = [[1, 0.25], [1, 0.89], [0, 0.93], [0, 0.48], [1, 0.19], [1, 0.49], [0, 0.6], [0, 0.6], [1, 0.34], [1, 0.19]],1,0.48
# split_on_numerical(d_set,a,sval) == ([[1, 0.25], [1, 0.19], [1, 0.34], [1, 0.19]],[[1, 0.89], [0, 0.93], [0, 0.48], [1, 0.49], [0, 0.6], [0, 0.6]])
# d_set,a,sval = [[0, 0.91], [0, 0.84], [1, 0.82], [1, 0.07], [0, 0.82],[0, 0.59], [0, 0.87], [0, 0.17], [1, 0.05], [1, 0.76]],1,0.17
# split_on_numerical(d_set,a,sval) == ([[1, 0.07], [1, 0.05]],[[0, 0.91],[0, 0.84], [1, 0.82], [0, 0.82], [0, 0.59], [0, 0.87], [0, 0.17], [1, 0.76]])
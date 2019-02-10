from random import shuffle
from ID3 import *
from operator import xor
from parse import parse
import matplotlib.pyplot as plt
import os.path
from pruning import *
import numpy as np
from numpy import *

# NOTE: these functions are just for your reference, you will NOT be graded on their output
# so you can feel free to implement them as you choose, or not implement them at all if you want
# to use an entirely different method for graphing

def get_graph_accuracy_partial(train_set, attribute_metadata, validate_set, numerical_splits_count, depth):
    '''
    get_graph_accuracy_partial - Given a training set, attribute metadata, validation set, numerical splits count, and percentage,
    this function will return the validation accuracy of a specified (percentage) portion of the trainging setself.
    '''
    tree = ID3(train_set, attribute_metadata, numerical_splits_count, depth)
    accuracy = validation_accuracy(tree, validate_set)

    reduced_error_pruning(tree, train_set,validate_set)
    accuracy_prune = validation_accuracy(tree, validate_set)

    return accuracy, accuracy_prune

def get_graph_data(train_set, attribute_metadata, validate_set, numerical_splits_count, iterations, pcts, depth):
    '''
    Given a training set, attribute metadata, validation set, numerical splits count, iterations, and percentages,
    this function will return the averaged graph accuracy partials based off the number of iterations.
    '''
    splits_count = numerical_splits_count[:]
    sum_acc = 0
    sum_acc_prune = 0
    train_set_length = len(train_set)

    for i in range(iterations):
        # get partial set based on pcts
        start = int(random.uniform(0, 1-pcts) * train_set_length)
        end = start + int(pcts * train_set_length)
        partial_set = train_set[start:end]

        # calculate accuracy using partial_set
        acc, acc_prune = get_graph_accuracy_partial(partial_set, attribute_metadata, validate_set, numerical_splits_count, depth)
        # calculate sum of accuracy
        sum_acc += acc
        sum_acc_prune += acc_prune

        numerical_splits_count = splits_count[:]

    return sum_acc / iterations, sum_acc_prune / iterations

# get_graph will plot the points of the results from get_graph_data and return a graph
def get_graph(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, iterations, lower, upper, increment):
    '''
    get_graph - Given a training set, attribute metadata, validation set, numerical splits count, depth, iterations, lower(range),
    upper(range), and increment, this function will graph the results from get_graph_data in reference to the drange
    percentages of the data.
    '''
    splits_count = numerical_splits_count[:]
    
    train_set_length = len(train_set)
    X, Y, Z = [], [], []
    i = 0.05
    while i <= 1.0:
        acc, acc_prune = get_graph_data(train_set, attribute_metadata, validate_set, numerical_splits_count, iterations, i, depth)
        numerical_splits_count = splits_count[:]

        X.append(i*train_set_length)
        Y.append(acc)
        Z.append(acc_prune)
        i += 0.025

    plt.plot(X, Y, X, Z, 'k--')
    plt.show()
    
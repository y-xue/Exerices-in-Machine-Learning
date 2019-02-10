from node import Node
from ID3 import *
from operator import xor

# Note, these functions are provided for your reference.  You will not be graded on their behavior,
# so you can implement them as you choose or not implement them at all if you want to use a different
# architecture for pruning.

max_d = 0
node_d = Node()
label_d = 0


def reduced_error_pruning(root,training_set,validation_set):
    '''
    take the a node, training set, and validation set and returns the improved node.
    You can implement this as you choose, but the goal is to remove some nodes such that doing so improves validation accuracy.
    '''
    global max_d
    global node_d
    global label_d

    while True: 
        acc_before = validation_accuracy(root, validation_set)
        if reduced_helper(root, validation_set, root, validation_set, acc_before) == False:
            break
        node_d.label = label_d
        node_d.children = {}

        max_d = 0
        node_d = Node()
        label_d = 0
    pass

def reduced_helper(root, validation_set, nodetmp, valid, acc_before):
    global max_d
    global node_d
    global label_d
    update = False
    if len(valid) == 0:
        return update
    if nodetmp.label == None:
        nodetmp.label = mode(valid)
        acc_after = validation_accuracy(root, validation_set)
        if acc_after - acc_before > max_d: 
            update = True
            max_d = acc_after - acc_before
            node_d = nodetmp
            label_d = node_d.label
        nodetmp.label = None

        if nodetmp.is_nominal:
            tmpv = split_on_nominal(valid, nodetmp.decision_attribute)
            for i in nodetmp.children.keys():
                if tmpv.has_key(i) == False:
                    continue
                update = update or reduced_helper(root, validation_set, nodetmp.children[i], tmpv[i], acc_before)
        elif nodetmp.is_nominal == False:
            tmpv = split_on_numerical(valid, nodetmp.decision_attribute, nodetmp.splitting_value)
            for i in nodetmp.children.keys():
                if len(tmpv) == 0:
                    continue
                update = update or reduced_helper(root, validation_set, nodetmp.children[i], tmpv[i], acc_before)
    
    return update
    pass

# def reduced_error_pruning(root,training_set,validation_set):
#     '''
#     take the a node, training set, and validation set and returns the improved node.
#     You can implement this as you choose, but the goal is to remove some nodes such that doing so improves validation accuracy.
#     '''
#     pass
    # acc_before = validation_accuracy(root, validation_set)
    # acc_after = 0
    # if root.label == None:
    #     att = root.decision_attribute
    #     isn = root.is_nominal
    #     spv = root.splitting_value
    #     chi = root.children
    #     nam = root.name
        
    #     root.label = mode(validation_set)
    #     # root.decision_attribute = None
    #     # root.is_nominal = None
    #     # root.splitting_value = None
    #     # root.children = {}
    #     # root.name = None

    #     acc_after = validation_accuracy(root, validation_set)

    #     if acc_after > acc_before:
    #         print acc_after, ">", acc_before
    #         return root
    #     else:
    #         root.label = None
    #         # root.decision_attribute = att
    #         # root.is_nominal = isn
    #         # root.splitting_value = spv
    #         # root.children = chi
    #         # root.name = nam
            
    #         if root.is_nominal:
    #             tmpv = split_on_nominal(validation_set, root.decision_attribute)
    #             for i in root.children.keys():
    #                 root.children[i] = reduced_error_pruning(root.children[i], training_set, tmpv[i])
    #         else:
    #             tmpv =  split_on_numerical(validation_set, root.decision_attribute, root.splitting_value)
    #             for i in root.children.keys():
    #                 root.children[i] = reduced_error_pruning(root.children[i], training_set, tmpv[i])
    # print acc_after, "<", acc_before              
    # return root

def validation_accuracy(tree,validation_set):
    '''
    takes a tree and a validation set and returns the accuracy of the set on the given tree
    '''
    
    cnt = 0
    for vset in validation_set:
        if tree.classify(vset) == vset[0]:
            cnt += 1

    return cnt * 1.0 / len(validation_set)

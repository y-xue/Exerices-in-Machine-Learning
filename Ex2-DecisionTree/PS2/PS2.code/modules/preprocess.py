import random

def preprocess(train_set, attribute_metadata):
    datalength = len(train_set)
    obslength = len(train_set[0])
    for i in xrange(1, obslength):
        # in parse.py '?' has been turned be None
        if attribute_metadata[i]['is_nominal']:
            attr_hash0 = {}
            max_0 = 0
            new_attr0 = 0
            for j in xrange(0, datalength):
                if train_set[j][i] != None:
                    if attr_hash0.has_key(train_set[j][i]):
                        attr_hash0[train_set[j][i]] += 1
                    else:
                        attr_hash0[train_set[j][i]] = 1
                    if attr_hash0[train_set[j][i]] > max_0:
                        max_0 = attr_hash0[train_set[j][i]]
                        new_attr0 = train_set[j][i]

            for j in xrange(0, datalength):
                if train_set[j][i] == None:
                    train_set[j][i] = new_attr0
    return train_set
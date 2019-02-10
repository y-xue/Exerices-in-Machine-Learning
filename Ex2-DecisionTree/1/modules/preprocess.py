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
        else:
            misvalue0 = 0
            if i != 5 and i != 6:
                s = []
                for data in train_set:
                    if data[i] != None:
                        s.append(data[i])
                s.sort()
                l = len(s)
                if l % 2 == 1:
                    misvalue0 = s[l/2]
                else:
                    misvalue0 = (s[l/2] + s[l/2 - 1])/2

            # count0 = 0.0
            # acc0 = 0.0
            # for j in xrange(0, datalength):
            #     if train_set[j][i] != None:
            #         count0 += 1
            #         acc0 += train_set[j][i]
            # misvalue0 = 0
            # if count0 != 0:
            #     misvalue0 = acc0 / count0
            for data in train_set:
                if data[i] == None:
                    data[i] = misvalue0
            print misvalue0

    return train_set
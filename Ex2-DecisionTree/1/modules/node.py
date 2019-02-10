# DOCUMENTATION
# =====================================
# Class node attributes:
# ----------------------------
# children - a list of 2 if numeric and a dictionary if nominal.  
#            For numeric, the 0 index holds examples < the splitting_value, the 
#            index holds examples >= the splitting value
#
# label - is None if there is a decision attribute, and is the output label (0 or 1 for
#	the homework data set) if there are no other attributes
#       to split on or the data is homogenous
#
# decision_attribute - the index of the decision attribute being split on
#
# is_nominal - is the decision attribute nominal
#
# value - Ignore (not used, output class if any goes in label)
#
# splitting_value - if numeric, where to split
#
# name - name of the attribute being split on

class Node:
    def __init__(self):
        # initialize all attributes
        self.label = None
        self.decision_attribute = None
        self.is_nominal = None
        self.value = None
        self.splitting_value = None
        self.children = {}
        self.name = None

    def classify(self, instance):
        '''
        given a single observation, will return the output of the tree
        '''
        # Your code here
        if self.label != None:
            return self.label
        if self.is_nominal == True:
            if self.children.has_key(instance[self.decision_attribute]):
                return self.children[instance[self.decision_attribute]].classify(instance)
            else:
                return self.value
                # self.label = 1
                # return self.label
        elif instance[self.decision_attribute] < self.splitting_value:
            return self.children[0].classify(instance)
        else:
            return self.children[1].classify(instance)

        # temp = self
        # while temp.label == None :
        #     if temp.is_nominal:
        #         if temp.children.has_key(instance[temp.decision_attribute]):
        #             temp = temp.children[instance[temp.decision_attribute]]
        #         else :
        #             temp.label = 0
        #     else:
        #         if instance[temp.decision_attribute] < temp.splitting_value:
        #             temp = temp.children[0]
        #         else :
        #             temp = temp.children[1]
        # return temp.label

    # def classify_nominal(self, instance):
    #     if len(instance) == 1 or len(self.children) == 0:
    #         print 1
    #         return self.label
    #     print 2
    #     return self.children[instance[1]].classify(instance[1:])

    # def classify_numeric(self, instance):
    #     if len(instance) == 1 or len(self.children) == 0:
    #         print 3
    #         return self.label
    #     if instance[1] < self.splitting_value:
    #         print 4
    #         return self.children[0].classify(instance[1:])
    #     else:
    #         print 5
    #         return self.children[1].classify(instance[1:])

    def print_tree(self, indent = 0):
        '''
        returns a string of the entire tree in human readable form
        '''
        # Your code here
        return self.print_to_str("", indent)

    def print_to_str(self, output, indent):
        for i in range(indent):
            if i % 2 == 0:
                output += "|"
            else:
                output += " "
        if self.label == None:
            output = output + str(self.name) + "\n"
            indent += 2
            if self.is_nominal:
                for node in self.children.itervalues():
                    output = node.print_to_str(output, indent)
            else:
                output = self.children[0].print_to_str(output, indent)
                output = self.children[1].print_to_str(output, indent)
            return output
        return output + str(self.label) + "\n"

    def print_dnf_tree(self):
        '''
        returns the disjunct normalized form of the tree.
        '''
        path_list = []
        output = ""
        self.find_path("", path_list)
        for path in path_list:
            output = output + " v (" + path + ")\n"
        return output[3:]       # delete leading " v "

    def find_path(self, path, path_list):
        if self.label > 0:
            path_list.append(path[3:])  # delete leading " ^ "
        elif self.label == None:
            if self.is_nominal:
                for output_value, node in self.children.iteritems():
                    node.find_path(path + " ^ " + self.name + " = " + str(output_value), path_list)
            else:
                self.children[0].find_path(path + " ^ " + self.name + " < " + str(round(self.splitting_value,3)), path_list)
                self.children[1].find_path(path + " ^ " + self.name + " >= " + str(round(self.splitting_value,3)), path_list)


def check_classify():
    n0 = Node()
    n0.label = 1
    i = 0;
    if n0.classify([0, 1, 2]) == 1:
        print "Passed 1"
        i += 1
    else:
        print n0.classify([0, 1, 2])
        print "Failed 1"
    n1 = Node()
    n1.label = 0
    n = Node()
    n.label = None
    n.decision_attribute = 1
    n.is_nominal = True
    n.name = "You saw the attributes what do you think?"
    n.children = {1: n0, 2: n1}
    if n.classify([0, 2]) == 0:
        print "Passed 2"
        i += 1
    else:
        print n.classify([0, 2])
        print "Failed 2"
    if i == 2:
        print "All tests passed"
    else:
        print "Not all tests passed, look at classify"

    print n.print_dnf_tree()
    print n.print_tree()

check_classify()
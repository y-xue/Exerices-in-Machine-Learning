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
        elif instance[self.decision_attribute] < self.splitting_value:
            return self.children[0].classify(instance)
        else:
            return self.children[1].classify(instance)

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

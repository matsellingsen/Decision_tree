import numpy as np
import pandas as pd

 # https://www.geeksforgeeks.org/decision-tree/ <-- good overview-explanation

 #TODO 
 # 1. create train (STRUCTURE has been SET UP; NEED TO FIX BUGS)
    # 1. calc expected entropy for each attribute (DONE(?))<-- Validate correctness here somehow
    # 2. split on min.val^, i.e. create children nodes with belonging data for each value of the split-attribute. (DONE)
    # 3. repeat until children node is leaf node, i.e. all data has same class or max depth is reached. (DONE)
# 2. create classify() #<-- Implemented, not tested.
# 3. create test()



class Decision_tree: # Using ID3-algorithm to calcualte splitting
    def __init__(self, tr, te, label_name, max_depth=100):
        self.label = label_name
        self.train_x = tr
        self.test_x = te
        self.max_depth = max_depth
        self.root = Node(tr, label_name, max_depth,  0)         

    def train(self, node=None):

        if node is None: 
            node = self.root
        def select_split_attribute():
            def expected_entropy(attribute: pd.DataFrame , attr_name: str): #<-- attribute is a pandas column 
                
                total_examples = len(attribute)  
                unique_values = pd.unique(attribute[attr_name])
                value_counts =  pd.value_counts(attribute[attr_name])
                unique_labels = pd.unique(attribute[self.label])
             
                probs = list(map(lambda x: np.multiply(-sum(
                                                            list(map(lambda y: np.multiply(
                                                                                        (np.divide(len(attribute.loc[(attribute[attr_name]==x) & (attribute[self.label]==y)]), value_counts[x])),
                                                                                         np.log2(np.divide(len(attribute.loc[(attribute[attr_name]==x) & (attribute[self.label]==y)]), value_counts[x]), where= np.divide(len(attribute.loc[(attribute[attr_name]==x) & (attribute[self.label]==y)]), value_counts[x])!=0)),
                                                            unique_labels))),
                                                        np.divide(value_counts[x], total_examples)
                                                        ),
                            unique_values))
               
                # ^ Calculating probability (H^i) for every value of the given attribute. (Note to self: See lecture on trees in Methods in AI)
                return sum(probs) #<-- expected entropy


            exp_entropy_attributes = list(map(lambda x: (expected_entropy(node.data[[x, self.label]], x), x), node.data.drop(columns=[self.label]).columns)) #<-- Calculating expected entropy for all attributes.
            split_attribute = min(exp_entropy_attributes, key = lambda x: x[0]) #<-- Finding attribute to split/branch, attribute with smallest expected entropy is selected.
            return split_attribute[1] #<-- Selected split attribute.

        if node.is_leaf: #<-- we have reached an end of the tree
            return 
        
        split_attribute = select_split_attribute()
        node.split_attribute = split_attribute
        groups = node.data.groupby([split_attribute])
        
        for value in pd.unique(node.data[split_attribute]): #Branching out by splitting data on unique values for selected split_attribute
            splitted_data = groups.get_group(value)
            splitted_data = splitted_data.drop(columns=[split_attribute])
            new_node = Node(splitted_data , self.label, self.max_depth, node.depth+1, split_attribute, value, node)   
            self.train(new_node)
            node.children.append(new_node)

    
    def classify(self, instance: pd.DataFrame):
        current_node = self.root

        def walk_tree(node: Node):
            if node.is_leaf:
                return node.label
            next_node = next(node for node in node.children if node.attribute_value == instance[node.split_attribute])
            walk_tree(next_node)


        """Intuitive explanation of what classify() does."""
        # Look at current node
        #   Find correct class
        # Update current node to leaf-node with correct class
        # If node== leaf, return majority class as answer
        # else: repeat  
        """"""

        return walk_tree(current_node)
             
        
    def test(self):
        "placeholder"


 
class Node:  
    def __init__(self, data, label_name, max_depth, depth, split_attribute=None, attribute_value = None, parent=None):
        self.label = label_name
        self.max_depth = max_depth
        self.parent = parent
        self.attribute_value = attribute_value
        self.data = data
        self.children = []
        self.depth = depth
        self.split_attribute = split_attribute
        
        if len(pd.unique(data[self.label])) == 1 or self.max_depth == depth or len(data.columns) == 2: #<-- Checking if node is a leaf-node.
            self.is_leaf = True
        else:    
            self.is_leaf = False



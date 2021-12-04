##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: dt.py
# 
# Description: A file containing a number of functions used for Decision Tree labeling
# of data.
##################################################################################
import numpy as np

class TreeNode():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index # Feature index for the conditional node
        self.threshold = threshold # Threshold value for the conditional node to split into the left and right child
        self.left = left # Left child
        self.right = right # Right child 
        self.info_gain = info_gain # Information gain based on the threshold value and feature index selected
        self.value = value # Value for leaf node


class DTClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        
        # Stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        """
        Builds the decision tree recursively by getting the best split based on the GINI value.

        Parameters:
            dataset: Training dataset appended with its label
            curr_depth: Current depth of the tree
      
        Returns:
            TreeNode: The decision node or leaf node of the tree
        """
        X, Y = dataset[:,:-1], dataset[:,-1] # Training data and corresponding label
        num_samples, num_features = np.shape(X)
        
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth and num_samples > 0: # Until the min sample limit/max depth stopping conditions are hit
            best_split = self.get_best_split(dataset, num_features) # Get the best split node as a dictionary
            # check if information gain is positive
            if best_split["info_gain"]>0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                return TreeNode(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        leaf_value = self.calculate_leaf_value(Y) # Get the leaf node value
        return TreeNode(value=leaf_value)
    
    def get_best_split(self, dataset, num_features):
        """
        Retrieves the best split point for growing the tree by maximizing the information gain. 

        Parameters:
            dataset: Training dataset appended with its label
            num_features: Total number of features present (value of k)
      
        Returns:
            TreeNode: Returns a dictionary with corresponding values for the best split on top of which the Tree is expanded.
        """
        best_split = {}
        # Initialize values to avoid crashes
        best_split["feature_index"] = None
        best_split["threshold"] = 0
        best_split["dataset_left"] = []
        best_split["dataset_right"] = []
        best_split["info_gain"] = 0
        max_info_gain = -float("inf")
        
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index] # Getting the feature values from dataset
            possible_thresholds = np.unique(feature_values) # All the unique values present in the dataset.
            for threshold in possible_thresholds:
                # split the dataset on each possible unique value
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # If the child nodes are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    print
                    # Updating the best split
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
         
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        """
        Splits the dataset based a certain threshold value for a selected feature

        Parameters:
            dataset: Training dataset appended with its label
            feature_index: Feature selected for doing the split on
            threshold: Threshold value for the split
      
        Returns:
            Split Dataset: Returns the split dataset
        """
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child):
        """
        Splits the dataset based a certain threshold value for a selected feature

        Parameters:
            dataset: Training dataset appended with its label
            feature_index: Feature selected for doing the split on
            threshold: Threshold value for the split
      
        Returns:
            Split Dataset: Returns the split dataset
        """
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        return gain
    
    def gini_index(self, y):
        # Compute the GINI index
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        # From all the candidate classes, select the maximum value as the final leaf value
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        # Concat the training data and the label
        dataset = np.concatenate((X, Y), axis=1) 
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        # Predict and classify the input data
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        
        if tree.value!=None: 
          return tree.value
        
        feature_val = x[tree.feature_index]
        
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
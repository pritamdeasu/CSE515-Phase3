##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: graph.py
# 
# Description: A file containing various graph functions used for PPR labeling
# in Tasks 1-3. The functions here have multiple 'versions' programmed in for
# testing.
##################################################################################

import numpy as np
import math
import itertools
from progressbar import progressbar

def get_transition_matrix(A, use_alt_version=False):
    """
    Creates a transition matrix for a graph of objects. Note that the probabilities in this
    transition matrix represent the normalized inverse distance values (so nodes closer together
    are significantly more likely to be reached).

    The transition matrix is formed by taking each row of the adjacency matrix, transposing and
    scaling, then setting it to the column of the transition matrix.

    This might need to be changed to a version which has equal probabilities to visit each
    node! Test this thoroughly!

    Parameters:
        A: Adjacency matrix representing the graph. Weights represent how close the objects are.
        use_alt_version: Switch to use "classic" version, where there is an equal weight to transition
            to each adjacent node.
    
    Returns:
        T: Transition matrix of the graph. Probabilities add up to 1 for each column.
    """
    T = np.zeros(A.shape)
    for i in range(A.shape[0]):
        if not use_alt_version:
            T[:, i] = A[i] / np.sum(A[i])
        else:
            indices = np.nonzero(A[i])
            T[indices, i] = 1 / np.count_nonzero(A[i])
    return T

def get_weighted_adj_matrix(input_image_dict, classify_image_dict, label_type, vector_distance_function, top_n=None, cutoff=None):
    """
    Creates a weighted adjacency matrix for an input set of images, where the weight on each
    edge is the INVERSE of the distance function used. This adjacency matrix is further processed
    to retain a certain number of connections and form a graph (stored in the adjacency matrix).

    In the adjacency matrix, images from the input directory are listed FIRST, then images from
    the directory to be classified. This data is stored in an index list to make things straightforward.

    At least one of the top_n and cutoff parameters must be specified so the graph is not super
    connected. If both are specified, then the graph will do the lowest n below the cutoff.
    
    If neither are specified, the adjacency matrix leaves all weights. Use the weighted transition
    matrix if you are doing this.

    Parameters:
        input_image_dict: Input image dictionary containing the images used to help classify.
        classify_image_dict: Dictionary of images to be classified.
        classify_parameter: Specifies whether to use type, subject, or index as the label type.
        vector_distance_function: Distance function used when computing the distance between two vectors.
            This is a FUNCTION and may depend on how the latent semantics were created.
        top_n: Optional parameter. Specify this if you want the graph to have at most n connections from each node.
        cutoff: Optional parameters. Specifies the largest possible weight for an edge.

    Returns:
        A: Adjacency matrix representing the graph.
        index list: An array of tuples of the form '(image_name, type, label)'
            Type is 0 (from input_dict) or 1 (from classify_dict). The label is either
            the correct label for the image or 'None'.
    """
    num_elements = len(input_image_dict.keys()) + len(classify_image_dict.keys())
    A = np.zeros((num_elements, num_elements))
    vector_list = [None] * num_elements
    tuple_list = [None] * num_elements
    
    # First, construct a temp array holding all feature vectors for easy processing
    # and set up the tuples
    current_index = 0
    for key, value_dict in input_image_dict.items():
        vector_list[current_index] = value_dict['latent']
        tuple_list[current_index] = (key, 0, value_dict[label_type])
        current_index += 1
    for key, value_dict in classify_image_dict.items():
        vector_list[current_index] = value_dict['latent']
        tuple_list[current_index] = (key, 1, None)
        current_index += 1
    
    # Now we have set up our tuples with information and a temporary list. We can construct
    # the adjacency matrix now.
    
    # Creating the adjacency matrix takes some time, so we added a progress bar
    for i in progressbar(range(num_elements)):
        for j in range(num_elements):
            if i != j:
                distance = vector_distance_function(vector_list[i], vector_list[j])
                if distance == 0:
                    distance += 0.0001
                A[i, j] = distance
        # Once we are done with a row, we then keep only the top n results
        # We do this by getting the LARGEST num_elements - n in each row then setting them to 0
        if top_n is not None:
            A[i][np.argpartition(A[i], top_n)[top_n:]] = 0
        # If the value is too high, set it to 0
        if cutoff is not None:
            A[i][np.where(A[i] > cutoff)] = 0
    
    return A, tuple_list
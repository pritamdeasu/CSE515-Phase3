##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: ppr.py
# 
# Description: A file containing a number of functions used for PPR labeling
# of data.
# 
# How do we implement a classifier based on PPR?
# Use the paper by Lin and Cohen for this
# Essentially, it boils down to the following:
#   - Create a graph of ALL nodes (labeled and unlabeled) with distances being
#        similarity/distance or something like that.
#   - Run PPR for each labeling class used, with the following parameters:
#     - If a node is labeled as class c, then use it in the 'teleport' vector
#     - Create the PPR here.
#   - Next, iterate through the unlabeled images, and assign them a cluster/label as follows:
#     - If the unlabeled instance has the highest PPR score for class c, then assign the instance to class c.
##################################################################################

import numpy as np
from lib import svd, pca, lda, kmeans, graph, util
from . import io   # This is necessary due to file naming. Oops
from progressbar import progressbar

def PPR_Classifier(input_image_dict, classify_image_dict, label_type, damping_parameter, dim_reduction_method, top_n=None, cutoff=None):
    """
    Function which performs PPR Classification on unlabeled data in classify_image_dict
    based on the labels in input_image_dict. This classifies images based on 3 different
    types of labels: 'subject', 'type', or 'id'.
    
    Parameters:
        input_image_dict: Input dictionary of labeled images from the database.
        classify_image_dict: Input dictionary of unlabeled images to be classified
        label_type: Specifies what we are classifying on. Should be 'type', 'subject', or 'id'
        damping_parameter: Value specifying the damping parameter (how often the surfer 'jumps')
        dim_reduction_method: Used to specify distance function used.
        top_n: Optional parameter to specify options when generating the adjacency matrix.
        cutoff: Optional parameter to specify options when generating the adjacency matrix.
    
    Returns:
        output_tuple_list: A list of tuples of the form:
            '(image_name, label)'. This is the labels assigned to all unlabeled images.
    """
    num_labeled_images = len(input_image_dict.keys())
    num_unlabeled_images = len(classify_image_dict.keys())
    output_tuple_list = [None] * num_unlabeled_images
    
    # First, select the distance function to use
    if dim_reduction_method == 'svd':
        distance_function = svd.vector_distance
    elif dim_reduction_method == 'pca':
        distance_function = pca.vector_distance
    elif dim_reduction_method == 'lda':
        distance_function = lda.vector_distance
    elif dim_reduction_method == 'kmeans':
        distance_function = kmeans.vector_distance
    
    print("Creating adjacency matrix...")
    A, tuple_list = graph.get_weighted_adj_matrix(input_image_dict, classify_image_dict, label_type, distance_function, top_n=top_n, cutoff=cutoff)
    print("Creating transition matrix...")
    T = graph.get_transition_matrix(A, use_alt_version=True)
    
    # Select the iterable based on the label type
    
    # In the case of using a different data set for labels, we need to ensure we
    # only classify on existing labels (in case we only have some image types/subjects/ids)
    label_iter = util.get_possible_classes(input_image_dict, label_type)
    
    """
    # TODO: Change this to only use 'used' labels. Maybe a set?
    if label_type == 'type':
        label_iter = io.TYPE
    elif label_type == 'subject':
        label_iter = range(1,41)
    elif label_type == 'id':
        label_iter = range(1,11)
    """
    
    # Step through the classes and generate PPR results for each class
    print("Computing PPR results for the input classes...")
    PPR_results = np.zeros((len(label_iter), A.shape[0]))
    # Create a progress bar, then compute the PPR results.
    for i in progressbar(range(len(label_iter))):
        seed_vector = create_seed_vector(tuple_list, label_iter[i])
        PPR_results[i] = getPageRank(T, seed_vector, damping_parameter)
    
    # Now, iterate through all unlabeled images and determine the label by finding the maximum score in a column
    for j in range(num_labeled_images, len(tuple_list)):
        output_tuple_list[j - num_labeled_images] = (tuple_list[j][0], label_iter[np.argmax(PPR_results[:, j])])
        
    # TODO: Determine unified format to return results. For now we will return the tuples corresponding to the 
    # images we originally wanted to label
    return output_tuple_list, label_iter
        
def create_seed_vector(tuple_list, label):
    """
    Creates seed vector for PPR classification. This is done by creating a seed vector where all indices
    labeled nodes with the correct class/label are assigned a value of 1/# nodes, and all other nodes
    are 0.
    
    Parameters:
        tuple_list: List of tuples indexing the PPR Classification. Should be generated when creating the adjacency matrix.
        label: The label on which the seed vector is created.
    
    Returns:
        A seed vector which can be used in PPR.
    """
    output = np.zeros(len(tuple_list))
    indices = [idx for idx, element in enumerate(tuple_list) if element[1] == 0 and element[2] == str(label)]
    output[indices] = 1 / len(indices)
    return output

def getPageRank(T, s_n, d):
    """
    Gets the PPR scores when given an input Transition matrix, jump matrix s, and
    damping parameter d.

    Parameters:
        T: The transition matrix.
        s_n: A vector representing probabilities of jumps. The 3 seeds are given probabilities of 1/3, all others are 0
        d: The probability to jump, rather than move to another node. This is the damping value.

    Returns:
        The PPR scores for all nodes in the graph.
    """
    I = np.identity(T.shape[0])
    temp = I - (1 - d)*T
    # If the matrix is not singular, we can directly solve for r. Otherwise, approximate
    return np.linalg.solve(temp, d*s_n) if np.linalg.det(temp) != 0 else np.linalg.lstsq(temp, d*s_n)

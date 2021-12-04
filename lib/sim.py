##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: sim.py
# 
# Description: A file containing various useful utility functions used on the
# data for the project. This includes transforming data dictionaries into arrays,
# extracting subject/type weight pairs, and constructing the subject-subject or
# type-type similarity matrices.
##################################################################################

import numpy as np
import collections
from scipy.sparse import data
from sklearn.metrics.pairwise import cosine_similarity

def get_type_subject_similarity_matrix(image_dict, feature_len, model, subject_bool = "False"):
    """
    Gets all the values of a subject/type, combines their feature vector values grouped by subject/type. We take average of all the values to get this value.
    The cosine similarity is calculated for this matrix and the resulted similarity matrix is returned

    Parameters:
        image_dict: A dictionary containing various information about the images processed. Should be a dict processed
            by io.read_images. 
        feature_len: Length of the feature model selected.
        model : The feature model selected 
        subject_bool: A bool to specify whether to compute subject-subject similarity matrix or type-type similarity matrix. If this is True,
            the function computes subject-subject similarity matrix. Otherwise, it computes type-type similarity matrix.
    
    Returns:
        N*N similarity matrix where N is equal to the number of subjects for the subject-subject similarity or type otherwise.
    """
    temp_matrix, index_array = compute_average_vectors(image_dict, feature_len, model, subject_bool)
    similarity_matrix = cosine_similarity(temp_matrix) #Cosine similarity calculated
    return similarity_matrix, index_array

def compute_average_vectors(image_dict, feature_len, model, subject_bool = "False"):
    """
    Computes the average feature vectors from input based on subject or type. Used to compute similarity matrices
    and when creating feature vectors when using latent semantics generated from tasks 3-4 in tasks 5-7.
    """
    key = "type" # Value for the key to find element in the image_dict
    
    if subject_bool == True:
        key = "subject"

    keywise_vals = {} # Stores the summation of feature values for all the subject(s)/type(s)
    counts = {} # Stores the count of the number of values for each of the subject/types

    for filename, value in image_dict.items():
        if value[key] in keywise_vals:
            counts[value[key]] += 1
            keywise_vals[value[key]] += value[model]
        else:
            keywise_vals[value[key]] = value[model]
            counts[value[key]] = 1
    
    if subject_bool == True:
        od = collections.OrderedDict(sorted(keywise_vals.items(), key=lambda x: int(x[0]))) # Sorted by key for subject
    else:
        od = collections.OrderedDict(sorted(keywise_vals.items())) # Sorted by key for type
    
    temp_matrix = np.empty((len(keywise_vals.keys()), feature_len), dtype=np.float64)

    index_array = [None] * len(od.items())
    i = 0
    for key, value in od.items():
        value /= counts[key] #Dividing the values by the total count to average
        temp_matrix[i] = value
        index_array[i] = key
        i += 1
    
    return temp_matrix, index_array

def vector_to_sim_space(avg_vectors, input_vector):
    """
    Converts an image feature vector into the image similarity space. Essentially, this compares an input vector
    to the "average" feature vector computed over each subject or type. Then, a feature array is formed such that
    entry i in the output feature vector is how similar the input vector is to the average subject/type 'i' vector.
    """
    # Note that we manually implement the cosine similarity here to avoid memory issues caused by skimage's implementation
    # of cosine similarity.
    return [np.dot(avg_vectors[i], input_vector)/(np.linalg.norm(avg_vectors[i])*np.linalg.norm(input_vector)) for i in range(len(avg_vectors))]
    # return [cosine_similarity(avg_vectors[i].reshape(-1, 1), input_vector.reshape(-1, 1)) for i in range(len(avg_vectors))]

def array_to_sim_space(avg_vectors, data_array):
    """
    Converts an array of feature vectors (where each row is a feature vector (color, ELBP, or HOG) of an image)
    into the similarity space.
    """
    output_array = [None] * len(data_array)
    for i in range(len(data_array)):
        output_array[i] = vector_to_sim_space(avg_vectors, data_array[i])
    return(output_array)

##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: util.py
# 
# Description: A file containing various useful utility functions used on the
# data for the project. This includes transforming data dictionaries into arrays,
# extracting subject/type weight pairs, and constructing the subject-subject or
# type-type similarity matrices.
##################################################################################

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial.distance

def feature_dict_to_array(input_dict, model):
    """
    Utility function which compresses feature vectors from an input image dictionary into a matrix for processing.

    Parameters:
        input_dict: An input dictionary where the keys are filenames and the values are dictionaries with information
            about each image (feature vector, subject, type, etc.)
        model: The feature model used to compute the feature vectors ('color', 'elbp', 'hog', or 'latent').

    Returns:
        An array where feature vectors are stacked into an array row-wise in the SAME order as images in the feature dictionary.
    """
    # First, we need to get the length of a vector
    vector_size = len(list(input_dict.values())[0][model])
    output = np.empty((len(input_dict), vector_size), dtype=np.float64)
    i = 0
    for filename, vector in input_dict.items():
        output[i] = vector[model]
        i += 1
    return output

def reject_outliers(data, m = 2.):
    """
    Function which rejects outlying data based on the median and standard deviation
    of the data. Note that the input 'data' MUST be a Numpy array! This function may not be necessary but
    is included if needed.

    Credit to user 'shaneb' at https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    for this function.

    Parameters:
        data: Numpy array with data, should be a vector.
        m: Value which affects the cutoff of the excluded data. Higher values increases tolerance for outliers.

    Returns:
        Numpy array with outlying data filtered given the below method.
    """
    #d = np.abs(data - np.median(data))
    #mdev = np.median(d)
    #s = d/mdev if mdev else 0.
    #return data[s<m]
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def compute_vector_difference_of_feature(feature_dict, compare_vector, feature):
    """
    Computes the distance between an input vector and all feature vectors of a certain type stored
    in an image dictionary (read from a folder)
    To determine the distance for the color moment feature descriptor, Euclidean distance is used.
    For ELBP and HOG, Earth Mover's distance is used.
    
    Parameters:
        feature_dict: Dictionary in which all feature data and ID values are stored.
        compare_id: Feature vector (original or latent) compared to dataset.
        feature: Feature to compare. This should be either 'color', 'elbp', 'hog', or 'latent',
            depending on the situation.
    
    Returns:
        A sorted list of tuples, with the id and the image name.
    """
    output_tuple_list = []
    # Compares the distance between all feature vectors except the original photo, since the distance
    # here would always be 0.
    for img_id, features in feature_dict.items():
        # If using ELBP or HOG, use Earth Mover's distance to compare images
        # If using Color moment or latent features, use Euclidean distance
        #if feature == 'elbp':
        #    output_tuple_list.append(img_id, compute_EM_distance(compare_vector, features[feature], 10, 640))
        #elif feature == 'hog':
        #    output_tuple_list.append(img_id, compute_EM_distance(compare_vector, features[feature], 36, 1764))
        #else:
        output_tuple_list.append((img_id, scipy.spatial.distance.euclidean(compare_vector, features[feature])))
    return output_tuple_list

def compute_EM_distance(vector_1, vector_2, hist_size, vector_size):
    """
    Computes the Earth Mover's Distance between two input vectors of equal length.
    For the purposes of this project, multiple feature descriptors (ELBP and HOG) use
    histograms to store values. The EM distance (or Wasserstein distance) computes how
    'difficult' it is to convert between the histograms. The sum of the EM distance
    for all histograms in a feature vector is returned as the distance value.
    Parameters:
        vector_1: The first input vector. This should be a 1-dimensional array.
        vector_2: The second input vector. This should be a 1-dimensional array equal in size to the first.
        hist_size: The size of histograms in the array.
        vector_size: The length of the input vectors.
    Returns:
        The sum of the EM distances for all input histograms. Smaller values mean the input vectors
        are more similar.
    """
    distance_arr = []
    # Resizes the input vectors to be the lists of histograms.
    hist_list_1 = np.resize(vector_1, (vector_size // hist_size, hist_size))
    hist_list_2 = np.resize(vector_2, (vector_size // hist_size, hist_size))
    # Computes the EM distance for each pair of histograms and returns the sum.
    for (hist_1, hist_2) in zip(hist_list_1, hist_list_2):
        distance_arr.append(scipy.stats.wasserstein_distance(hist_1, hist_2))
    return np.sum(distance_arr)

def get_possible_classes(input_image_dict, label_type):
    """
    Gets the possible labels which can be assigned to input images in Tasks 1-3.
    """
    label_set = set()
    for key, value in input_image_dict.items():
        label_set.add(value[label_type])
    return sorted(label_set)

def convert_svm_results(comparison_image_dict, predicted_labels, potential_labels):
    """
    Simple helper function to transform results from SVM in Tasks 1-3 into a consistent
    format for printing. This is used for integration.
    """
    output_tuple_list = []
    i = 0
    for key in comparison_image_dict:
        output_tuple_list.append((key, potential_labels[predicted_labels[i]]))
        i += 1
    return(output_tuple_list)

def convert_dt_results(comparison_image_dict, predicted_labels, potential_labels):
    """
    Simple helper function to transform results from DT in Tasks 1-3 into a consistent
    format for printing. This is used for integration.
    """
    output_tuple_list = []
    i = 0
    for key in comparison_image_dict:
        output_tuple_list.append((key, potential_labels[predicted_labels[i]]))
        i += 1
    return(output_tuple_list)

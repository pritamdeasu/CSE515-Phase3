##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: pca.py
# 
# Description: A file containing functions useful for the PCA dimensionality
# reduction technique. This was implemented for this project, not used from
# an external library.
##################################################################################

import numpy as np
import collections

def pca(A,k):
    """
    Computes the PCA dimensional reduction on an input data matrix. Various Numpy methods are used to
    help this process (such as to compute covariance arrays or compute eigenvalues) but the PCA itself
    is implemented here.

    Parameters:
        A: Input data matrix A containing the feature vectors of the images.
        k: Number of latent semantics to select.

    Returns:
        A matrix that is the result of the PCA reduction, with the k latent semantics selected.
    """
    # mean Centering the data  
    A_meaned = A - np.mean(A , axis = 0)
    cov_mat = np.cov(A_meaned , rowvar = False)
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)

    #sort the eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
 
    sorted_eigenvalue = eigen_values[sorted_index]
    #similarly sort the eigenvectors 
    sorted_eigenvectors = eigen_vectors[:,sorted_index]

    eigenvector_subset = sorted_eigenvectors[:,0:k]

    A_reduced = np.dot(eigenvector_subset.transpose(), A_meaned.transpose()).transpose()
    return A_reduced,eigenvector_subset

def extract_weight_pairs(U, image_dict, k, subject_bool):
    """
    Selects the subject/type-weight pairs from the result of the PCA decomposition.

    Parameters:
        U: The input array
        image_dict: A dictionary of image values
        k: The number of latent semantics
        subject_bool: A boolean indicating whether to select subjects or types.
    """
    output_list = [None] * k
    for i in range(k):
        j = 0
        temp_list = []
        for filename in image_dict:
            temp_str = str(image_dict[filename]['subject']) if subject_bool else image_dict[filename]['type']
            temp_list.append((temp_str, U[j, i]))
            j += 1
        
        temp_dict = collections.defaultdict(list)
        for x1, x2 in temp_list:
            temp_dict[x1].append(x2)

        output_list[i] = sorted([(x1, sum(x2, 0.0) / len(x2)) for x1, x2 in temp_dict.items()], key=lambda x: x[1], reverse=True)
    return output_list
    
def extract_sim_weight_pairs(U, index_array, k):
    """
    Returns a list of tuples from the result of dimensionality reduction via SVD. This is used when using similarity matrices (since there
    is no need to aggregate values).

    Parameters:
        U: Truncated U array produced by the SVD algorithm.
        index_array: A list of indices matched to subjects/types for the pairs.
        k: The number of latent semantics chosen.
    
    Returns:
        An output list, where each row corresponds to a list of sorted subject/type-weight pairs.
    """
    return [sorted([(index_array[j], U[j, i]) for j in range(len(index_array))], key=lambda x: x[1], reverse=True) for i in range(k)]


def map_vector(P, q):
    """
    Uses method discussed in https://nlp.stanford.edu/IR-book/essir2011/pdf/18lsi.pdf to remap an input
    feature vector to a latent feature space.

    Essentially, maps the new vector by multiplying:
        q * V * Σ^(-1)
    to give a new vector in terms of the k latent features.

    Note that we normalize this vector!

    Parameters:
        sigma: Matrix Σ from the result of SVD computation.
        V: The V matrix from the SVD computation.
        q: Feature vector to be mapped to the latent semantic space.

    Returns:
        Feature vector described in terms of the k latent semantics.
    """
    q_k = q @ P
    return q_k / np.linalg.norm(q_k)

def map_all_vectors(P, vector_array):
    """
    A function which maps ALL vectors read from an image database into the latent feature space created
    by SVD. This iteratively performs the map_vector() function above.
    """
    output_array = [None] * len(vector_array)
    for i in range(len(vector_array)):
        output_array[i] = map_vector(P, vector_array[i])
    return np.array(output_array)

def find_n_most_similar(U, q_k, n, filenames):
    """
    Finds the n most similar images to input vector q_k from an SVD feature decomposition.

    TODO: Potentially implement different distance measures.

    Parameters:
        U: The U matrix from an SVD run on some data. Should be size m x k, where m is the number of images and k is the number
            of latent features selected.
        q_k: The input image vector compared to vectors from U.
        n: The number of images to return.
        filenames: An ordered list of filenames corresponding to vectors in U.

    Returns:
        A sorted list of n tuples of the form '(filename, distance)'. This can be used to display the n most similar images.
    """
    output_list = [None] * U.shape[0]
    for i in range(U.shape[0]):
        # Currently, the Euclidean distance between vectors is computed.
        output_list[i] = (filenames[i], vector_distance(U[i], q_k))
    return sorted(output_list, key=lambda x: x[1])[:n]

def similarity_classification(U, q_k, image_dict, subject_bool):
    """
    When provided a matrix of feature vectors mapped to the latent feature space, as well as
    an input vector q_k (also mapped to the latent feature space), finds the most similar subject/type
    to the image given under the current latent semantics. At the moment, this is done by computing
    differences then averaging the resulting values.

    Parameters:
        U: Input array with images mapped to the feature space (either original or new).
        q_k: Our comparison vector mapped to the latent feature space.
        image_dict: An image dictionary containing some information. Importantly, the keys in this dictionary are
            sorted by their order in the rows of U!
        subject_bool: True if the user wishes to find the most similar subject, False if the user wishes to find the most similar type.

    Returns:
        A string indicating the subject/type, depending on the input parameters, as well as the average value.
    """
    temp_dict = {}
    # Switch determines what key is used in the dictionary.
    switch = 'subject' if subject_bool else 'type'
    i = 0
    # Iterate through the filenames and array U to create lists of distance values
    for filename, value_dict in image_dict.items():
        if value_dict[switch] not in temp_dict:
            temp_dict[value_dict[switch]] = []
        temp_dict[value_dict[switch]].append(vector_distance(U[i], q_k))
        i += 1
    
    # After we create the dictionary of lists of distance values, convert the dictionary elements to averages of the distances.
    for key in temp_dict:
        temp_dict[key] = np.mean(temp_dict[key])

    # We then get the minimum value from this dictionary, which is the smallest average distance, and return the key
    min_key = min(temp_dict, key=temp_dict.get)
    return min_key, temp_dict[min_key], temp_dict

def vector_distance(vector_1, vector_2):
    """
    Computes the vector distance between 2 vectors. This function exists to be a common function for computing
    distance vectors mapped to the SVD feature space (so it is easy to change the distance function for all
    relevant functions, if necessary).
    """
    return np.linalg.norm(vector_1 - vector_2)

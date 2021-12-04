##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: lda.py
# 
# Description: A file containing functions useful for the LDA dimensionality
# reduction technique used throughout the project.
##################################################################################

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation
import collections
import numpy as np

def lda(data_array, k):
    """
    Performs the Latent Dirichlet Analysis (LDA) on an input data array and returns the 'k' most significant
    latent semantics. The LDA implementation from sklearn is used for this, as we were allowed to use existing
    libraries for this dimensionality reduction technique.

    Parameters:
        data_array: Input data array on which the LDA is computed. Should be the feature vectors for the selected images.
        k: Number of latent semantics to select.

    Returns:
        The resulting n*k LDA matrix.
        LDA components as k*n matrix
    """
    # Create LDA object
    lda_model = LatentDirichletAllocation(n_components=k, learning_method='online')

    # Normalize the data extracted to values between zero and one and multiply this result by 100 to be able to convert it into discrete values
    data_array = (data_array-data_array.min())/(data_array.max()-data_array.min())*100

    # Round off all the values in the matrix to closest integer value
    for i in range(len(data_array)):
        for j in range(len(data_array[0])):
            data_array[i][j]=round(data_array[i][j])

    # Get the fitted and transformed output from the library
    lda_matrix = lda_model.fit_transform(data_array)
    
    # LDA Components values, Used in tasks 5 through 7 for mapping vectors to the latent space
    lda_components=  lda_model.components_
    
    return lda_matrix, lda_components

def extract_weight_pairs(U, image_dict, k, subject_bool):
    """
    Extracts the subject/type-weight pairs for the selected dimensionality reduction technique(PCA/SVD/LDA). Requires a matrix U, where the rows of U are the entries in
    the image dictionary in order. The columns are the k latent semantics discovered by the selected technique. In order to compute
    the subject-weight pairs, the values for each subject are averaged for each latent semantic.

    It is crucial that the order of the input dictionary is NOT changed between reading of images and computation
    of the reduction technique; otherwise the rows of the matrix U will not correspond to the correct subject/type.

    Parameters:
        U: The 'U' matrix that is the output of the above SVD function.
        image_dict: A dictionary containing various information about the images processed. Should be a dict processed
            by io.read_images. 
        k: The number of latent semantics computed.
        subject_bool: A bool to specify whether to compute subject-weight pais or type-weight pairs. If this is True,
            the function computes subject-weight pairs. Otherwise, it computes type-weight pairs.
    
    Returns:
        An output list. The ith element in the output list is a list of sorted subject/type-weight pairs for
        latent semantic i. This can directly be saved to a file by functions in io.py.
    """
    output_list = [None] * k
    for i in range(k):
        j = 0
        temp_list = []
        for filename in image_dict:
            temp_str = str(image_dict[filename]['subject']) if subject_bool else image_dict[filename]['type']
            temp_list.append((temp_str, U[j, i]))
            j += 1
        # Now, we compute the average based on the first term in the array of tuples
        temp_dict = collections.defaultdict(list)
        for x1, x2 in temp_list:
            temp_dict[x1].append(x2)

        # This line is pretty busy, but what it does is compute the average value for each subject/type for a given
        # latent semantic, then place this ordered set of tuples in the corresponding output list element.
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

def map_vector(lda_components, vector):
    """
    Maps a target vector to the latent space generated by LDA. In this case, the LDA components is
    a 'k x m' array which describes each latent feature in terms of the m old features. We can use matrix
    multiplication to map the vector to the latent feature space.
    """
    return vector @ np.transpose(lda_components)

def map_all_vectors(lda_components, vector_array):
    """
    Maps all vectors in an array to the latent feature space generated by LDA.
    """
    return np.array([map_vector(lda_components, vector_array[i]) for i in range(len(vector_array))])

def find_n_most_similar(D, vector, n, filenames):
    """
    Finds the n most similar images to input vector q_k from an SVD feature decomposition.

    Parameters:
        D: A data array containing feature vectors of all images mapped to the latent feature space.
        q_k: The input image vector compared to vectors from U.
        n: The number of images to return.
        filenames: An ordered list of filenames corresponding to vectors in U.

    Returns:
        A sorted list of n tuples of the form '(filename, distance)'. This can be used to display the n most similar images.
    """
    output_list = [None] * D.shape[0]
    for i in range(D.shape[0]):
        # Currently, the Euclidean distance between vectors is computed.
        output_list[i] = (filenames[i], vector_distance(D[i], vector))
    return sorted(output_list, key=lambda x: x[1])[:n]

def similarity_classification(D, vector, image_dict, subject_bool):
    """
    When provided a matrix of feature vectors mapped to the latent feature space, as well as
    an input vector q_k (also mapped to the latent feature space), finds the most similar subject/type
    to the image given under the current latent semantics. At the moment, this is done by computing
    differences then averaging the resulting values.

    TODO: Implement a more robust similarity test (such as the one proposed here: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list)

    TODO: Implement a similar function which works with the semantics from Tasks 3-4???

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
        temp_dict[value_dict[switch]].append(vector_distance(D[i], vector))
        i += 1
    
    # After we create the dictionary of lists of distance values, convert the dictionary elements to averages of the distances.
    # TODO: Better distance metric and way to filter outlying data
    for key in temp_dict:
        #temp_dict[key] = util.reject_outliers(np.array(temp_dict[key]))
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

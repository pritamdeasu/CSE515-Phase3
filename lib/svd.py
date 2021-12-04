##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: svd.py
# 
# Description: A file containing functions useful for the SVD dimensionality
# reduction technique. This was implemented for this project, not used from
# an external library.
#
# The functions here are identical to those used in Phase 2 of the project.
##################################################################################

import numpy as np
import collections
from lib import util

def get_random_unit_vector(length):
    """
    Computes a random unit vector of a certain length. Used for the power method
    to compute the SVD of a matrix. Uses Numpy to get normally distribted values
    in the range (0, 1] then normalizes the vector.

    Parameters:
        length: The length of the unit vector to compute.

    Returns:
        A 1-dimensional unit vector with 'length' elements.
    """
    random_vec = np.random.rand(length).astype(np.float64)
    return random_vec / np.linalg.norm(random_vec)

def compute_largest_sv(A, epsilon = 1e-10, debug = False):
    """
    Computes the largest singular value, as well as the corresponding left and right singular
    vectors. Uses an iterative power method to compute the value.

    Credit to the following resources for information on the SVD power method:
        https://www.cs.yale.edu/homes/el327/datamining2013aFiles/07_singular_value_decomposition.pdf
        https://jeremykun.com/2016/05/16/singular-value-decomposition-part-2-theorem-proof-algorithm/
    
    Parameters:
        A: Input matrix on which the largest singular value is computed.
        epsilon: "Error" value to determine when the singular vector is "good enough".
                Defaults to 1e-10, but can be changed if desired. May be useful to decrease
                if too many iterations are encountered.
        debug: An optional flag to enable some debug print statements.

    Returns:
        u_1: Left singular vector corresponding to the largest singular value.
        sigma_1: Largest singular value for matrix A.
        v_1: Right singular vector corresponding to the largest singular value.
    """
    n, m = A.shape
    current_vector = get_random_unit_vector(m)
    prev_vector = np.zeros(m)
    ATA = A.T @ A
    compare_value = 1 - epsilon
    
    i = 0
    # We run this loop until the dot product of the two vectors is within the error range.
    # That is, we want the dot product to be in the range (1-ε, 1].
    while abs(current_vector @ prev_vector) <= compare_value:
        prev_vector = current_vector
        current_vector = ATA @ prev_vector
        current_vector = current_vector / np.linalg.norm(current_vector)
        i += 1

    # According to the first reference, computes the largest singular vectors and
    # singular value. Note that v_1 is already normalized due to how the above loop runs.
    v_1 = current_vector
    Av_1 = A @ v_1
    sigma_1 = np.linalg.norm(Av_1)
    u_1 = Av_1 / sigma_1

    if debug:
        print(f"Found v_1 in {i} iterations!")
        #print(f"v_1: {v_1}\nsigma_1: {sigma_1}\nu_1: {u_1}")
    
    return u_1, sigma_1, v_1

def svd(A, k = None, epsilon = 1e-10, debug = False):
    """
    Computes the SVD of a given input matrix using the power method, which performs a set of random steps to find 
    right singular vector v, then the corresponding singular value and left singular vector. This is done iteratively,
    finding the largest singular value each loop, until all singular values have been found. This function is not
    especially efficient, especially on large matrices (such as 1000x1000); however, it is difficult to implement more
    efficient computation of the SVD without the use of significantly more complex algorithms.

    Due to the random walk required, this algorithm will take a long time to run if values are very similar and sizes
    grow much larger than 600-700 elements. Please tweak input parameters if the calculation takes too long.

    This algorithm will only compute the top k latent features if a value of k is specified.

    Parameters:
        A: Input matrix on which the SVD is calculated.
        k: Optional parameter to specify how many singular values to compute. NOTE: Currently broken, do not set.
        epsilon: Parameter to specify the minimum amount of error required. Do NOT reduce this value when using very
            large matrices; it may be useful to increase this value if computation is taking too long.
        debug: An optional flag which enables some debug print statements.
    
    Returns:
        A set of matrices U, Σ, V such that A = UΣV^T. Note that U and V are returned as expected, but Σ is returned
        as a vector of singular values in decreasing order. It can be transformed into the correct diagonal matrix Σ
        (of size n x m) if necessary.
    """
    n, m = A.shape
    # Sets k to m by default. Also sets k to m if an invalid input is given (such as k < 1 or k > m).
    if k == None or k > m or k < 1:
        k = m

    # Initialize our vectors. Note that we will create our 'sigma' matrix later, since
    # we can simply diagonalize the vector of all singular values.
    U = np.empty((n, k))
    singular_values = np.empty(k)
    V = np.empty((k, m))

    # We create a temporary matrix, and run the loop 'm' times to find all singular values.
    # Note that due to how the power method works, we subtract the value of sigma*(u @ v.T) from
    # the A matrix in each loop to find the next singular value.
    temp_matrix = A.copy().astype(np.float64)
    for i in range(k):
        u, sigma, v = compute_largest_sv(temp_matrix, epsilon, debug)
        U[:, i] = u
        singular_values[i] = sigma
        V[i] = v
        temp_matrix -= sigma * np.outer(u, v)

    if debug:
        print(f"U:\n{U}\nSigma:\n{singular_values}\nV:\n{V.T}")

    return U, singular_values, V.T

def shape_sigma(singular_values, shape):
    """
    A helper function which reshapes an array of singular values into a correct shape. This is a diagonal matrix
    with potentially more rows/columns, depending on the shape of the original matrix A.

    Parameters:
        singular_values: A Numpy array containing all singular values.
        shape: A tuple containing the desired shape. Should be of the form '(n, m)'.

    Returns:
        A reshaped diagonal array containing the singular values. If the shape was correct, this is precisely
        the array Σ in a SVD of some matrix A.
    """
    output = np.zeros(shape)
    for i in range(len(singular_values)):
        output[i, i] = singular_values[i]
    return output

def extract_weight_pairs(U, image_dict, k, subject_bool):
    """
    Extracts the subject/type-weight pairs for SVD. Requires a matrix U, where the rows of U are the entries in
    the image dictionary in order. The columns are the k latent semantics discovered by SVD. In order to compute
    the subject-weight pairs, the values for each subject are averaged for each latent semantic.

    It is crucial that the order of the input dictionary is NOT changed between reading of images and computation
    of the SVD; otherwise the rows of the matrix U will not correspond to the correct subject/type.

    Parameters:
        U: The 'U' matrix that is the output of the above SVD function.
        image_dict: A dictionary containing various information about the images processed. Should be a dict processed
            by io.read_images. 
        k: The number of latent semantics computed.
        subject_bool: A bool to specify whether to compute subject-weight pairs or type-weight pairs. If this is True,
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

def map_vector(sigma, V, q):
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
    q_k = q @ V @ np.linalg.inv(sigma)
    return q_k / np.linalg.norm(q_k)

def map_all_vectors(sigma, V, vector_array):
    """
    A function which maps ALL vectors read from an image database into the latent feature space created
    by SVD. This iteratively performs the map_vector() function above.
    """
    output_array = [None] * len(vector_array)
    for i in range(len(vector_array)):
        output_array[i] = map_vector(sigma, V, vector_array[i])
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

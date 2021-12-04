##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: feature.py
# 
# Description: A file containing library functions used to compute the color
# moment, ELBP, and HOG feature descriptors on images.
#
# The code here is directly taken from Aaron Steele's implementation of the
# feature descriptors for Phase 1 of the project.
##################################################################################

import numpy as np
import math
import itertools
import skimage.feature
import itertools

def compute_mean(data):
    """
    Function which computes the mean of a slice of data. Note that this only works on data passed
    in as greyscale, and the data must be pre-sliced!
    
    Parameters:
        data: Slice of data to compute the mean on. Should be a Numpy array.
    
    Returns:
        The mean of the data provided.
    """
    return np.sum(data) / data.size

def compute_stddev(data, mean = None):
    """
    Function which computes the standard deviation of a slice of data. Note that this only works on data passed
    in as greyscale, and the data must be pre-sliced!
   
    Parameters:
        data: Slice of data to compute standard deviation on. Should be a Numpy array.
        mean: Optional argument to provide the mean if it is already computed. 'None' by default.
    
    Returns:
        The standard deviation of the data provided.
    """
    # Computes mean if not provided, but does not recompute if not necessary
    if mean is None:
        mean = compute_mean(data)
    # Creates and applies function to all elements in array, then performs the stddev operation
    f = lambda x: (x - mean)**2
    return math.sqrt(np.sum(f(data)) / data.size)

def compute_skewness(data, mean = None):
    """
    Function which computes the skewness of a slice of data. Note that this only works on data passed
    in as greyscale, and the data must be pre-sliced!
    
    Parameters:
        data: Slice of data to compute skewness on. Should be a Numpy array.
        mean: Optional argument to provide the mean if it is already computed. 'None' by default.
    
    Returns:
        The skewness of the data provided.
    """
    # Computes mean if not provided, but does not recompute if not necessary
    if mean is None:
        mean = compute_mean(data)
    # Creates and applies function to all elements in array, then performs the skewness operation
    f = lambda x: (x - mean)**3
    return np.cbrt(np.sum(f(data)) / data.size)
    # return scipy.stats.skew(data.flatten())

def calculateSkewness(xi, mu, std):
    """
    Alternate function to compute the skewness.

    Parameters:
        xi: The slice of data passed in.
        mu: The mean of the data.
        std: The standard deviation of the data.
    
    Returns:
        The skewness of the input data.
    """
    numerator=0
    for i in range(len(xi)):
        for j in range(len(xi[i])):
            numerator+=(xi[i][j]-mu)**3
    return np.nan_to_num((numerator/((xi.shape[0]*xi.shape[1]-1)*(std**3))),0)


def get_color_moments(image, windowsize = (8, 8)):
    """
    Function which gets the color moments of an input image, which is passed in a Numpy array, in greyscale,
    with values in the range [0, 1].
    
    Parameters:
        image:      Numpy array of the image itself.
        windowsize: Tuple representing the desired size of windows over which to compute the color moments.
                        Note that each parameter must be a divisor of the corresponding parameter in the
                        input image. Default for this phase is (8, 8).
    Returns:
        A Numpy array, which contains the color moments for mean, standard deviation, then skewness of
        of the data computed in order. Note that results are listed in row-major order for the windows of
        the input image (so all color moments for mean are listed in row-major order, then standard deviation,
        then skewness).
    """
    # Basic error checking to ensure that the window size will work evenly with the shape. Also checks
    # to ensure that the input image is a 2D array as expected.
    (window_x, window_y) = windowsize
    if image.ndim != 2 or image.shape[0] % window_x != 0 or image.shape[1] % window_y != 0:
        print("Image has too many dimensions or the window size is not correct!")
        exit(0)
    # Compute the number of windows and create our output arrays
    num_windows_x = int(image.shape[0] / window_x)
    num_windows_y = int(image.shape[1] / window_y)
    mean_arr = np.zeros((num_windows_x, num_windows_y))
    stddev_arr = np.zeros((num_windows_x, num_windows_y))
    skewness_arr = np.zeros((num_windows_x, num_windows_y))
    
    # Iterates over each slice, computing all 3 of the color moments
    for i, j in itertools.product(range(num_windows_x), range(num_windows_y)):
        slice = image[i*window_x:(i+1)*window_x, j*window_y:(j+1)*window_y]
        mean = compute_mean(slice)
        mean_arr[i, j] = mean
        stddev_arr[i, j] = compute_stddev(slice, mean)
        skewness_arr[i, j] = compute_skewness(slice, mean)
        #skewness_arr[i, j] = calculateSkewness(slice, mean, stddev_arr[i, j])
    return np.concatenate((mean_arr, stddev_arr, skewness_arr), axis=None)

def compute_ELBP(image, radius = 1, windowsize = (8, 8)):
    """
    Function which computes the Extended Local Binary Patterns model on an input image. In particular,
    the Local Binary Patterns are computed, and then a non-rotational uniform extension is computed on
    the data to reduce the feature vector size. This is done with skimage's implementation of LBP and
    the non-rotational uniform extension.
    
    After ELBP data is computed for each pixel in the image, the image is broken up into chunks and normalized
    histograms are computed for each section. These are then flattened in row-major order to create a feature vector.
    
    Parameters:
        image: The image to be processed
        radius: The radius of the ELBP to compute. For this project, this should not be a value other than 1.
        windowsize: The size of the windows over which to compute the ELBP histograms. This should be a tuple,
            and the values should be multiples of the first and second dimensions of the image respectively.
    
    Returns:
        A flattened Numpy array of size numWindows*10 (so 640 with the default settings).
        Each set of 10 elements corresponds to a normalized histogram
        of the non-rotational uniform ELBP of a window in the image, flattened in row-major order.
    """
    # Note that since we are using the NON-rotational invariant uniform extension of the LBP, there are
    # 59 bins into which data goes. Either it does not describe a uniform pattern (and is labeled as '0')
    # or it fits into one of the 58 uniform patterns and is labeled as such.
    NUMHISTBINS = 10
    # Basic error checking to ensure that the window size will work evenly with the shape. Also checks
    # to ensure that the input image is a 2D array as expected.
    (window_x, window_y) = windowsize
    if image.ndim != 2 or image.shape[0] % window_x != 0 or image.shape[1] % window_y != 0:
        print("Image has too many dimensions or the window size is not correct!")
        exit(0)
    # Compute the number of windows and create our output arrays
    num_windows_x = int(image.shape[0] / window_x)
    num_windows_y = int(image.shape[1] / window_y)
    # We split the image into cells, and compute the ELBP over each cell with skimage's implementation.
    # Then, a histogram is computed, normalized, and saved for each region.
    # NOTE: the ELBP is computed over the ENTIRE image (so pixels at the edge of cells
    #       can get more accurate texture values). The histograms are computed over the cells.
    lbp = skimage.feature.local_binary_pattern(image, 8 * radius, radius, 'uniform')
    out_array = []
    for i, j in itertools.product(range(num_windows_x), range(num_windows_y)):
        # This computes the normalized histogram, where the bins are the integers from 0 to 58.
        hist = np.histogram(lbp[i*window_x:(i+1)*window_x,j*window_y:(j+1)*window_y], bins=NUMHISTBINS, range=(0, NUMHISTBINS), density=True)[0]
        out_array.append(hist)
    return np.array(out_array).flatten()

def compute_HOG(image, num_orientations = 9, num_pixels_per_cell = (8, 8), num_cells_per_block = (2, 2), default_block_norm = 'L2-Hys'):
    """
    Function which computes the Histograms of oriented gradients (HOG) of an input image. The function is placed here for
    convenience and consistency of input arguments. Note that with default arguments, this will only work with Olivetti dataset
    images (or other 64x64 greyscale images). Parameters are those recommended for the Project, as some trial and error has
    shown these settings to produce the most useful output.
    
    Parameters:
        image: Image on which the HOG is computed.
        num_orientations: Number of orientation bins used in the HOG.
        num_pixels_per_cell: A tuple representing the number of pixels per cell in the HOG.
        num_cells_per_block: A tuple representing the number of cells per block in the HOG.
        default_block_norm: The block norm to use with the HOG function. This should probably not be changed.
    
    Returns:
        A numpy array which contains the HOG. This is our HOG feature vector for the input image.
        This is 7*7*2*2*9 = 1764 elements long.
    """
    return skimage.feature.hog(image, orientations=num_orientations, pixels_per_cell=num_pixels_per_cell, cells_per_block=num_cells_per_block, block_norm=default_block_norm)

def compute_features(input_dict, model):
    """
    Computes feature vectors on images passed in an input dictionary. Will
    compute the feature vectors for color, ELBP, or HOG, then place these into
    a dictionary which is returned.

    Parameters:
        input_dict: Dictionary of input images. Images should be read by IO functions and
                    stored with the key being the filename.
        model: Model to use. Should be 'color', 'ELBP', or 'HOG
    
    Returns:
        A dictionary with feature vectors referenced by filename.
    """
    output = {}
    temp = np.empty
    for filename, image_dict in input_dict.items():
        image = image_dict['image']
        # Changed so all feature vectors are always computed
        """
        if model == 'color':
            temp = get_color_moments(image)
        elif model == 'elbp':
            temp = compute_ELBP(image)
        elif model == 'hog':
            temp = compute_HOG(image)
        input_dict[filename][model] = temp.astype(np.float64)
        """
        temp = get_color_moments(image)
        input_dict[filename]['color'] = temp.astype(np.float64)
        temp = compute_ELBP(image)
        input_dict[filename]['elbp'] = temp.astype(np.float64)
        temp = compute_HOG(image)
        input_dict[filename]['hog'] = temp.astype(np.float64)
        
        # Feature_len is now HARDCODED
        if model == 'color':
            feature_len = 192
        elif model == 'elbp':
            feature_len = 640
        else:
            feature_len = 1764
    return feature_len

def compute_single_feature(input_image, model):
    """
    Computes a single feature vector on an input image.

    Parameters:
        input_dict: Dictionary of input images. Images should be read by IO functions and
                    stored with the key being the filename.
        model: Model to use. Should be 'color', 'ELBP', or 'HOG'.
    
    Returns:
        A feature vector corresponding to the chosen model.
    """
    if model == 'color':
        return get_color_moments(input_image)
    elif model == 'elbp':
        return compute_ELBP(input_image)
    elif model == 'hog':
        return compute_HOG(input_image)
    else:
        print("Invalid option chosen!")
        exit(0)
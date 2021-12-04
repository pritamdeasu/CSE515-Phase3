##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: latent.py
# 
# Description: A file containing a number of functions used for computing latent
# semantics on the data. Used for organizational purposes.
##################################################################################

from lib import io, pca, svd, kmeans, lda, feature, util

def compute_latent_semantics(input_folder, model, dim_red_technique, k, output_folder, training):
    """
    Computes latent semantics on all images in a given folder based on a certain
    dimensionality reduction method. Also saves the results to an output folder.

    Parameters:
        input_folder: Input folder from which images are read.
        model: Model used for feature extraction (CM, ELBP, or HOG)
        dim_red_technique: Dimensionality reduction method used to get latent semantics
        k: The number of latent semantics to compute
        output_folder: The output folder to which the latent semantic file is written (so future iterations
            do not need to redo these expensive calculations)
    
    Returns:
        image_dict: A dictionary of images read into the program with feature vectors computed
    """
    image_dict = io.read_images(input_folder)
    feature_len = feature.compute_features(image_dict, model)
    
    # Feature_len is now HARDCODED
    if model == 'color':
        feature_len = 192
    elif model == 'elbp':
        feature_len = 640
    else:
        feature_len = 1764

    # Second check to ensure k is not too large (this depends on the feature selected)
    if k > feature_len:
        print(f"k is too large for the selected feature length!\nWhen using model {model} ensure that k is not larger than {feature_len}.")
        exit(1)
    
    # Computes the dictionary of filenames and image feature vectors into a matrix which can be used for data processing
    data_array = util.feature_dict_to_array(image_dict, model)

    # NOTE: As of Python 3.7+, dictionaries remain in the same order. We can use
    # this to our 'advantage' (since feature vectors are added to the data array row-wise in order).
    #print(data_array[9] == image_dict['image-original-10-1.png'][model])

    # We run different dimensionality reduction techniques depending on the model. All outputs are saved to a 'results'
    # variable which is saved to the latent semantics file (and useful for tasks 5-7)
    if dim_red_technique == 'pca':
        A, P = pca.pca(data_array,k)
        results = [A,P]
    elif dim_red_technique == 'svd':
        U, singular_values, V = svd.svd(data_array, k=k, debug=False)
        results = [U, svd.shape_sigma(singular_values, (k, k)), V]
    elif dim_red_technique == 'lda':
        lda_matrix, lda_components = lda.lda(data_array, k)
        results = [lda_matrix, lda_components]
    elif dim_red_technique == 'kmeans':
        latent_semantics,latent_semantics_all = kmeans.kMeansAlgo(model, data_array, [*image_dict.keys()], k)
        results = [latent_semantics,latent_semantics_all]
    
    map_image_dict_to_latent(image_dict, model, dim_red_technique, results)
    if(training==True):
        io.write_latent_semantics(output_folder, [model, dim_red_technique, k, image_dict, results], model, dim_red_technique, k)
    return image_dict, results

def map_image_dict_to_latent(image_dict, model, dim_red_technique, results):
    """
    Maps a large number of image feature vectors to the latent feature space. Used when initially creating
    the image dictionary.

    Parameters:
        image_dict: Image dictionary used to store data about input images
        model: Feature vector model used
        dim_red_technique: Dimensionality reduction model used to generate the latent features
        results: The results from creation of latent features
    """
    for key, value_dict in image_dict.items():
        # Get the feature vector
        #print(value_dict)
        vector = value_dict[model]
        # Map the results depending on the dimensionality reduction method used to create the features.
        if dim_red_technique == 'pca':
            [A, P] = results
            image_dict[key]['latent'] = pca.map_vector(P,vector)
        elif dim_red_technique == 'svd':
            # We map the vector into the space of the latent semantics, then compare it to all U vectors to determine
            # which images are most similar
            [U, sigma, V] = results
            image_dict[key]['latent'] = svd.map_vector(sigma, V, vector)
        elif dim_red_technique == 'lda':
            [lda_matrix, lda_components] = results
            image_dict[key]['latent'] = lda.map_vector(lda_components, vector)
        elif dim_red_technique == 'kmeans':
            [latent_semantics,latent_semantics_file] = results
            #TODO: Method to JUST map latent features to a kmeans space
            print("Error! Map_image_dict_to_latent currently does NOT work with kmeans!")
            exit(1)

def map_single_image_to_latent(input_image_vector, dim_red_technique, results):
    """
    Maps a single image feature vector to a corresponding latent space.
    """
    if dim_red_technique == 'pca':
        [A, P] = results
        out_vector = pca.map_vector(P, input_image_vector)
    elif dim_red_technique == 'svd':
        # We map the vector into the space of the latent semantics, then compare it to all U vectors to determine
        # which images are most similar
        [U, sigma, V] = results
        out_vector = svd.map_vector(sigma, V, input_image_vector)
    elif dim_red_technique == 'lda':
        [lda_matrix, lda_components] = results
        out_vector = lda.map_vector(lda_components, input_image_vector)
    elif dim_red_technique == 'kmeans':
        [latent_semantics,latent_semantics_file] = results
        #TODO: Method to JUST map latent features to a kmeans space
        print("Error! Map_image_dict_to_latent currently does NOT work with kmeans!")
        exit(1)
    return out_vector

def compute_latent_sematics_from_data_arr(data_array, dim_red_technique = 'pca', k = 10):
    """
    Maps data array directly to latent space (Used for Task 6)
    """
    if dim_red_technique == 'pca':
        A, P = pca.pca(data_array,k)
        results = A
    elif dim_red_technique == 'svd':
        U, singular_values, V = svd.svd(data_array, k=k, debug=False)
        results = U
    # Not implemented for LDA and K-Means
    return results
##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: LSH.py
# 
# Description: A file to create the LSH index structure.
##################################################################################

import numpy as np
import math
from math import *
from operator import itemgetter

def LSH(hashes, data_array, layers, technique_result_query_image, image_dict, query_image_name):
    """
    Function which creates an LSH index and performs a query on the index structure. Used primarily
    in Task 4. The method used is one of dot products with random vectors ('hashes' vectors per layer)
    where the resulting bit of a hash is 1 if the dot product is >0, and 0 otherwise.
    
    Parameters:
        hashes: The number of hashes per layer.
        data_array: A processed data array where each row is a (selected) feature vector for the images
            to be stored in the LSH index structure.
        layers: The number of layers in the LSH.
        technique_result_query_image: The feature vector/latent feature computed for the input query image.
        image_dict: The input image dictionary. Used to get the names of images.
        query_image_name: The name of the query image.
    
    Returns:
        LSH: The LSH index structure.
        Query: The hashed query image to each layer.
        random_vector: The original hashing vector used to hash new feature vectors.
    """
    random_vector = [np.random.randn(hashes, len(data_array[0])) for j in range(layers)]
    LSH = []
    Query = []
    for layer in range(layers):
        layered_hash_bucket=dict()
        #j = 0
        for i,image in zip(data_array,image_dict.keys()):
            hash_value=random_vector[layer].dot(np.array(i))
            hashval="".join(['1' if i>0 else '0' for i in hash_value])
            #j = j + 1
            layered_hash_bucket.setdefault(hashval,[]).append(image)
        LSH.append(layered_hash_bucket)

        query_layered_hash_bucket=dict()
        # for i, image in zip(technique_result_query_image, [query_image]):
        hash_value = random_vector[layer].dot(np.array(technique_result_query_image))
        hash_value = "".join(['1' if i>0  else '0' for i in hash_value])
        query_layered_hash_bucket.setdefault(hash_value, []).append(query_image_name)
        Query.append(query_layered_hash_bucket)

    return LSH, Query, random_vector

def create_base_LSH(hashes, data_array, layers, image_dict):
    """
    A function used to create an LSH index structure, but not actually perform
    the query until later. Useful for Task 8.
    
    Parameters:
        hashes: The number of hashes per layer.
        data_array: A processed array of input images where each row corresponds to the
            (selected) feature vector in the image dictionary, in the same order.
        layers: The number of layers in the LSH structure.
        image_dict: The image dictionary. Used to get image names.
        
    Returns:
        lsh: The LSH index.
        hash_vector: The vector used to map new entries (such as a query image) to the LSH index.
    """
    # Creates a random vector. This is used for the dot product hashing.
    random_vector = [np.random.randn(hashes, len(data_array[0])) for j in range(layers)]
    lsh = []
    Query = []
    # Iterates through all layers
    for layer in range(layers):
        layered_hash_bucket=dict()
        # For a given layer, computes the hashes of all images in the image dictionary.
        for i,image in zip(data_array, image_dict.keys()):
            hash_value = random_vector[layer].dot(np.array(i))
            hashval = "".join(['1' if i>0 else '0' for i in hash_value])
            layered_hash_bucket.setdefault(hashval,[]).append(image)
        lsh.append(layered_hash_bucket)
    return lsh, random_vector

def map_LSH_query(layers, LSH_hash_vectors, query_image_vector, query_image_name):
    """
    Maps an LSH query image to an LSH hash space. The query needs to then be compared
    to the LSH index to find relevant images (for further processing).
    
    Parameters:
        layers: The number of layers in the LSH index structure.
        LSH_hash_vectors: The array of random vectors used in the LSH.
        query_image_vector: The feature/latent vector of the query image.
        query_image_name: The name of the query image.
        
    Returns:
        query: An array containing 'layer' dictionaries. The key is the important part
            here, as it contains the hash of the input image for each layer. Order
            is preserved (so DO NOT reorder the query result!)
    """
    query = []
    # Iterate through each layer and compute the hash of the query image for each layer.
    for layer in range(layers):
        query_layered_hash_bucket=dict()
        hash_value = LSH_hash_vectors[layer].dot(np.array(query_image_vector))
        hash_value = "".join(['1' if i>0  else '0' for i in hash_value])
        query_layered_hash_bucket.setdefault(hash_value, []).append(query_image_name)
        # print(query_layered_hash_bucket)
        query.append(query_layered_hash_bucket)
    return query

def get_relevant_images(Lsh, Query, layers, t):
    """
    Gets the relevant images in an LSH index structure. In particular, in each layer
    """
    relevant_images = []
    buckets = 0
    i = 0
    if len(list(set(relevant_images))) < t:
        while len(list(set(relevant_images))) < t:
            for layer in range(layers):
                search_key = list(Query[layer])[0]
                res = [val for key, val in Lsh[layer].items() if search_key[0:len(search_key)-i] == key[0:len(key)-i]]
                for j in res:
                    for k in  j:
                        relevant_images.append(k)
                        buckets = buckets + 2**i
            i = i + 1
            
    return relevant_images, buckets

def get_t_most_similar(lsh, query, query_image_vector, layers, t, image_dict, model):
    """
    Returns the t most similar images to a given query image by searching through an LSH hash structure.
    This is done by searching through the bucket in each layer which 'query' mapped to and finding
    all images in the bucket. Once all images are collected, the Euclidean distance is computed to all feature
    vectors and the 't' most similar images are returned.
    
    Parameters:
        lsh: The LSH index structure.
        query: An array containing the mapped query results.
        query_image_vector: The feature/latent feature vector of the query image.
        layers: The number of layers in the LSH structure.
        image_dict: The image dictionary storing feature vectors and whatnot.
        model: Specifies the feature vector/latent vector to use.
        
    Returns:
        An array containing the t most similar images (from the search) in decreasing order of importance.
    """
    relevant_images = []
    # Note that we search through additional buckets if we do not get 't' images after
    # the first iteration. 
    i = 0
    while len(list(set(relevant_images))) < t:
        for layer in range(layers):
            search_key = list(query[layer])[0]
            res = [val for key, val in lsh[layer].items() if search_key[0:len(search_key)-i] == key[0:len(key)-i]]
            for j in res:
                for k in j:
                    relevant_images.append(k)
            i += 1
    # Next, parse through all returned images and compute the distances to the query image
    distance_tuples = [(image_name, calculate_euclidean_distance(query_image_vector, image_dict[image_name][model])) for image_name in set(relevant_images)]
    distance_tuples.sort(key = itemgetter(1))
    # Return just the t closest image names
    return [image_name for (image_name, value) in distance_tuples]
    
def create_html(final,filename,t):
    # TEMP VARIABLES. CHANGE LATER!!!
    overall_images = 10
    unique_images = 10
    folder_path = "tmp"
    
    html_op = ("<html><head></head><body><h2><b>Output for Task 4</b></h2>")
    html_op += ("<h2>Overall Images %s </h2>" % (overall_images))
    html_op += ("<h2>Unique Images %s &nbsp; </h2>" % (unique_images))
    html_op += ("<h3>Top %s similar Images &nbsp; </h3>" % (t))
    html_op += ("<table>")

    i = 0
    for image in range(min(len(final),t)):
        if i % 6 == 0:
            html_op += ("<tr>")

        html_op += ("<td><div style='text-align:center'><img src='%s%s' width=200 height=200>ImageId:%s  Score:%f</div></td>" % (folder_path , final[image][0], final[image][0] , final[image][1]))
        if (i+1)%6==0:
            html_op+=("</tr>")
        i+=1
    html_op += ("</tr></table>")
    html_op += "</body></html>"

    file = open(filename,"w")
    file.write(html_op)
    
# Calculate Euclidean distance between two descriptors
def calculate_euclidean_distance(descriptor,test_descriptor):
    distance = math.sqrt(sum([(float(a)-float(b))**2 for a,b in zip(descriptor,test_descriptor)]))
    return distance
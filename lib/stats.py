##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: stats.py
# 
# Description: A file containing a number of functions used to compute statistics
# on results, such as the false positives/miss rates of data.
##################################################################################

from lib import io, util

def compute_statistics(input_tuples, classify_image_dict, label_iter, label_type):
    """
    Computes false positive and miss rates.
    
    Parameters:
        input_tuples: An input of tuples containing results from classification. Should
            be of the form ('key', 'label')
        classify_image_dict: The original image dictionary. Contains information necessary
            to determine whether or not a label was correct.
        label_type: A string indicating the type of label. This should be 'type', 'subject', or 'id'.
            
    Returns:
        A list of tuples corresponding to the labels. Each lists the corresponding false positive and miss rate,
        where each tuple is: ('label', FPR, MR)
    """
    # Note that we pass in label_iter, since we only want to consider possible labels.
    output_tuples = []
    for label in label_iter:
        output_tuples.append((str(label),
                              compute_FPR(input_tuples, classify_image_dict,label_type, str(label)),
                              compute_MR(input_tuples, classify_image_dict, label_type, str(label))))
    total_correct = 0
    for tup in input_tuples:
        if str(classify_image_dict[tup[0]][label_type]) == str(tup[1]):
            total_correct += 1
    return output_tuples, total_correct/len(input_tuples)
    
def compute_FPR(input_tuples, classify_image_dict, label_type, label):
    """
    Computes the False Positive Rate for classified input images. This is FP/(FP + TN).
    
    Parametes:
        input_tuples: A list of input tuples containing the labels assigned to the images.
        classify_image_dict: The original dictionary, contains the correct labels of input images.
        label_type: A string indicating the type of label. Should be 'type', 'subject', or 'id'.
        label: A string representing the class on which the false positive rate is computed.
        
    Returns:
        A value representing the false positive rate.
    """
    false_positives = 0
    true_negatives = 0
    for tup in input_tuples:
        if str(classify_image_dict[tup[0]][label_type]) != label and str(tup[1]) != label:
            true_negatives += 1
        elif str(classify_image_dict[tup[0]][label_type]) != label and str(tup[1]) == label:
            false_positives += 1
    return false_positives / (false_positives + true_negatives) if false_positives + true_negatives != 0 else 0
    
def compute_MR(input_tuples, classify_image_dict, label_type, label):
    """
    Computes the Miss Rate for classified input images. This is FN/(TP + FN).
    
    Parametes:
        input_tuples: A list of input tuples containing the labels assigned to the images.
        classify_image_dict: The original dictionary, contains the correct labels of input images.
        label_type: A string indicating the type of label. Should be 'type', 'subject', or 'id'.
        label: A string representing the class on which the miss rate is computed.
        
    Returns:
        A value representing the miss rate.
    """
    true_positives = 0
    false_negatives = 0
    for tup in input_tuples:
        if str(classify_image_dict[tup[0]][label_type]) == label and str(tup[1]) == label:
            true_positives += 1
        elif str(classify_image_dict[tup[0]][label_type]) == label and str(tup[1]) != label:
            false_negatives += 1
    return false_negatives / (true_positives + false_negatives) if true_positives + false_negatives != 0 else 0
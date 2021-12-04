##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: kmeans.py
# 
# Description: A file containing a number of functions used for the kmeans latent
# semantic extraction. This file is identical to the file used for submission
# in Phase 2 of the project.
##################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import scipy

def getDistanceHOG_ELBP(x,y):
    return scipy.spatial.distance.euclidean(x,y)

def getDistanceCM(x,y):
    return sum(abs(x-y))

def getDistanceSq(x,y):
    temp=x-y
    sum_sq=np.dot(temp.T,temp)
    return sum_sq

class KMeans:
    def __init__(self,model,k=5,max_iter=1000,plot_steps=False,strategy=2):
        self.k=k
        self.max_iter=max_iter
        self.plot_steps=plot_steps
        
        #list of indices for each cluster
        self.clusters=[[] for _ in range(self.k)]
        #mean feature for each cluster
        self.centroids=[]
        self.strategy=strategy
        self.model=model
        
    def predict_clusters(self,X):
        self.X=X
        self.n_Of_Samples,self.num_Of_Features=X.shape
        
        #initialize centroids for strategy2
        if (self.strategy==2):
            #initialize the first centroid
            random_indx=np.random.choice(self.n_Of_Samples,1,replace=False)
            self.centroids=[self.X[idx] for idx in random_indx]
            #remove the centroid from the sample list
            X1=np.delete(self.X,random_indx[0],0)
            #calculate rest of the centroids
            for i in range(0,self.k-1):
                #measure the distances and take argmax
                distance=[]
                for each in X1:
                    sum=0
                    for i in self.centroids:
                        if(self.model=='color'):
                            sum=sum+getDistanceCM(each,i)
                        if(self.model=='elbp' or self.model=='hog'):
                            sum=sum+getDistanceHOG_ELBP(each,i)
                    sum=sum/len(self.centroids)
                    distance.append(sum)
                nearest_index=np.argmax(distance)
                #set the ith centroid
                centroid_i=X1[nearest_index]
                self.centroids.append(centroid_i)
                X1=np.delete(X1,nearest_index,0)
            
                
        #initialize centroids for strategy1        
        if (self.strategy==1):
            random_indx=np.random.choice(self.n_Of_Samples,self.k,replace=False)
            self.centroids=[self.X[idx] for idx in random_indx]

        #random_indx=np.random.choice(self.n_Of_Samples,self.k,replace=False)
        #self.centroids=[self.X[idx] for idx in random_indx]
        #self.centroids=[[2,0],[2,2]]
        #optimization
        for _ in range(self.max_iter):
            #update clusters
            self.clusters=self.build_cluster(self.centroids)
            if self.plot_steps:
                self.graph_Plot()
            #update centroids
            centroid_old=self.centroids
            self.centroids=self.get_centroids(self.clusters) #assign mean value of the clusters to the centroid
            # print(self.centroids)
            if self.plot_steps:
                self.graph_Plot()
            #check if converged
            #if distances between old and new centroids do not change, then break
            if self.is_Converged(centroid_old,self.centroids):
                #print(self.centroids)
                #print("Converging after: ",_," iterations for ",self.k," clusters in strategy: "+str(self.strategy))
                break

        #return cluster labels
        return self.cluster_labels(self.clusters),self.centroids
    
    def cluster_labels(self,clusters):
        labels=np.empty(self.n_Of_Samples)
        for cluster_index,cluster_no in enumerate(clusters):
            for sample_index in cluster_no:
                labels[sample_index]=cluster_index
        return labels
        
    def build_cluster(self,centroids):
        clusters=[[] for _ in range(self.k)]
        for index,each_sample in enumerate(self.X):
            centroid_index=self.nearest_centroid(each_sample,centroids)
            clusters[centroid_index].append(index)
        return clusters
        
    def nearest_centroid(self,sample,centroids):
        #calculate distance of current sample to each centroid
        if self.model=='color':
            distances=[getDistanceCM(sample,pt) for pt in centroids]
        if self.model=='hog' or self.model=='elbp':
            distances=[getDistanceHOG_ELBP(sample,pt) for pt in centroids]
        #index with minimum distance
        nearest_index=np.argmin(distances)
        return nearest_index
    
    def get_centroids(self,clusters):
        #initialize centroids
        centroids=np.zeros((self.k, self.num_Of_Features))
        for cluster_index, cluster_no in enumerate(self.clusters):
            # A fix in the case when
            cluster_mean=np.nan_to_num(np.mean(self.X[cluster_no],axis=0), 0)
            centroids[cluster_index]=cluster_mean
        return centroids
    
    def is_Converged(self,centroid_old,centroids):
        if self.model=='color':
            distances=[getDistanceCM(centroid_old[i], centroids[i]) for i in range(self.k)]
        elif self.model=='hog' or self.model=='elbp':
            distances=[getDistanceHOG_ELBP(centroid_old[i],centroids[i]) for i in range(self.k)]
        return sum(distances)==0 #sum 0 means no change in centroids
    
    def graph_Plot(self):
        fig,axs=plt.subplots(figsize=(20,15))
        
        for i,index in enumerate(self.clusters):
            point=self.X[index].T
            axs.scatter(*point,s=60)
        
        for point in self.centroids:
            axs.scatter(*point,marker="x",color="black",linewidth=20,s=30)
         
        plt.show()
def getLatentSemantics_AllbyKmeans(model,features,filenames,K):
    '''
    Parameters
    ----------
    model: CM or HOG or ELBP
    features: extracted features (CM, HOG or ELBP)
    K : number of latent semantics
    filenames: list of files for creating subject-weight or type-weight pairs, for example: ['image-emboss-1-1.png', 'image-emboss-1-2.png', 'image-emboss-1-3.png']
    Description of the function: Computes K centroids and the distances from all points to those K centroids
    Returns
    -------
    Top K latent semantics of all images in the directory
    '''

    k_model = KMeans(model,K,max_iter=10000,plot_steps=False,strategy=2)
    y_pred,cluster_centers = k_model.predict_clusters(np.array(features))

    _weight_pair = OrderedDict()
    for each in filenames:
        _weight_pair[each]=[0]*K

    
    for i in range(len(filenames)):
        latent_semantics=[]
        for each in cluster_centers:
            if(model=='color'):
                distance = sum(abs(val1-val2) for val1,val2 in zip(each,features[i]))
                latent_semantics.append(distance)
            elif (model=='elbp' or model=='hog'):
                distance = scipy.spatial.distance.euclidean(each,features[i])
                latent_semantics.append(distance)
            
        _weight_pair[filenames[i]]=[item1+ item2 for (item1,item2) in zip(_weight_pair[filenames[i]],latent_semantics)]

    latent_semantics_all={}
    latent_semantics_all['Model']=model
    latent_semantics_all['Weight_Pair']=_weight_pair
    latent_semantics_all['Centroids']=cluster_centers
    return latent_semantics_all
def kMeansAlgo(model, features, filenames, K):
    '''

    Parameters
    ----------
    model: CM or HOG or ELBP
    features: extracted features (CM, HOG or ELBP)
    K : number of latent semantics
    filenames: list of files for creating subject-weight or type-weight pairs, for example: ['image-emboss-1-1.png', 'image-emboss-1-2.png', 'image-emboss-1-3.png']
    Description of the function: Computes K centroids and the distances from all points to those K centroids
    Returns
    -------
    Top K latent semantics

    '''
    k_model = KMeans(model,K,max_iter=10000,plot_steps=False,strategy=2)
    y_pred,cluster_centers = k_model.predict_clusters(np.array(features))
    
    _weight_pair = OrderedDict()
    
    for each in filenames:
        _weight_pair[each.split('-')[1]+'-'+each.split('-')[2]]=[0]*K
    
    for i in range(len(filenames)):
        k=filenames[i].split('-')[1]+'-'+filenames[i].split('-')[2]
        latent_semantics=[]
        for each in cluster_centers:
            if(model=='color'):
                distance = sum(abs(val1-val2) for val1,val2 in zip(each,features[i]))
            elif (model=='elbp' or model=='hog'):
                distance = scipy.spatial.distance.euclidean(each,features[i])
            latent_semantics.append(distance)

        _weight_pair[k]=[item1+ item2 for (item1,item2) in zip(_weight_pair[k],latent_semantics)]

        

        
        
    latent_semantics_set={}
    for i in range(K):
        temp_list=[]
        for each in _weight_pair.keys():           
            temp_list.append((each,_weight_pair[each][i]))            
        latent_semantics_set['Latent Semantic-'+str(i)]=temp_list

        '''
        As distance from centroids to all points are measured, less distance between the centroid and 
        the object means that the objects have more weight
        '''
    for each in latent_semantics_set.keys():
        latent_semantics_set[each]=sorted(latent_semantics_set[each],key=lambda x:x[1], reverse=True)
        
    # with open(outputFile,'w') as fp:
    #     json.dump(latent_semantics_set, fp, cls=NumpyEncoder,indent=1)
    
    return latent_semantics_set,getLatentSemantics_AllbyKmeans(model,features,filenames,K)

def getLatentSemantics_AllbyKmeansSim(model,features,similarity_matrix,K):
    #print(similarity_matrix)
    '''
    Parameters
    ----------
    model: CM or HOG or ELBP
    features: extracted features (CM, HOG or ELBP)
    K : number of latent semantics
    filenames: list of files for creating subject-weight or type-weight pairs, for example: ['image-emboss-1-1.png', 'image-emboss-1-2.png', 'image-emboss-1-3.png']
    Description of the function: Computes K centroids and the distances from all points to those K centroids
    Returns
    -------
    Top K latent semantics of all images in the directory
    '''

    k_model = KMeans(model,K,max_iter=10000,plot_steps=False,strategy=2)
    y_pred,cluster_centers = k_model.predict_clusters(np.array(features))

    _weight_pair = OrderedDict()
    for each in similarity_matrix:
        _weight_pair[each]=[0]*K

    
    for i in range(len(similarity_matrix)):
        k=similarity_matrix[i]
        latent_semantics=[]
        for each in cluster_centers:
            distance=0
            if(model=='color'):
                distance = sum(abs(val1-val2) for val1,val2 in zip(each,features[i]))
                
            elif (model=='elbp' or model=='hog'):
                distance = scipy.spatial.distance.euclidean(each,features[i])
            latent_semantics.append(distance)
            
        _weight_pair[k]=[item1+ item2 for (item1,item2) in zip(_weight_pair[k],latent_semantics)]

    
    latent_semantics_file={}
    latent_semantics_file['Model']=model
    latent_semantics_file['Weight_Pair']=_weight_pair
    latent_semantics_file['Centroids']=cluster_centers
    return latent_semantics_file

def kMeansAlgo_sim(model, features, similarity_matrix, K):
    '''

    Parameters
    ----------
    model: CM or HOG or ELBP
    features: extracted features (CM, HOG or ELBP)
    K : number of latent semantics
    filenames: list of files for creating subject-weight or type-weight pairs, for example: ['image-emboss-1-1.png', 'image-emboss-1-2.png', 'image-emboss-1-3.png']
    Description of the function: Computes K centroids and the distances from all points to those K centroids. A variation
        of the above function with some changes to work with similarity matrices.
    Returns
    -------
    Top K latent semantics

    '''
    k_model = KMeans(model,K,max_iter=10000,plot_steps=False,strategy=2)
    y_pred,cluster_centers = k_model.predict_clusters(np.array(features))
    
    _weight_pair = OrderedDict()
    
    for each in similarity_matrix:

        _weight_pair[each]=[0]*K

    
    for i in range(len(similarity_matrix)):
        k=similarity_matrix[i]
        latent_semantics=[]
        for each in cluster_centers:
            if(model=='color'):
                distance = sum(abs(val1-val2) for val1,val2 in zip(each,features[i]))
            elif (model=='elbp' or model=='hog'):
                distance = scipy.spatial.distance.euclidean(each,features[i])
            latent_semantics.append(distance)
        _weight_pair[k]=[item1+ item2 for (item1,item2) in zip(_weight_pair[k],latent_semantics)]
        
        
    latent_semantics_set={}
    for i in range(K):
        temp_list=[]
        for each in _weight_pair.keys():           
            temp_list.append((each,_weight_pair[each][i]))            
        latent_semantics_set['Latent Semantic-'+str(i)]=temp_list

        '''
        As distance from centroids to all points are measured, less distance between the centroid and 
        the object means that the objects have more weight
        '''
    for each in latent_semantics_set.keys():
        latent_semantics_set[each]=sorted(latent_semantics_set[each],key=lambda x:x[1], reverse=True)
        
    # with open(outputFile,'w') as fp:
    #     json.dump(latent_semantics_set, fp, cls=NumpyEncoder,indent=1)
    
    return latent_semantics_set,getLatentSemantics_AllbyKmeansSim(model,features,similarity_matrix,K)

def getKMeansPairs(latent_semantics, subject_bool):
    """
    Re-organizes the output of the above kmeans function into a standardized form which can be written to a file
    with io.write_pairs().

    Parameters:
        latent_semantics: Array of latent semantics generated by KMeans function above.
        subject_bool: True if we return the subject-weight pairs, False if we want the type-weight pairs.

    Returns:
        Standardized array of sorted subject/type-weight pairs.
    """
    index = 1 if subject_bool else 0
    return [[(f"{x1.split('-')[index]}", x2) for x1, x2 in value] for key, value in latent_semantics.items()]

def getKMeansPairs_sim(latent_semantics):
    return [[(x1, x2) for x1, x2 in value] for key, value in latent_semantics.items()]

def identifyTopN(vector,latent_semantics_file,data_array,query_image_name,K,n=1):
    '''
    Parameters
    -----------
    query_image_path: a list of full path of query image
    query_image_name: a list of query image name
    latent_semantics_file: input latent semantic
    given the filename (list) of a query image which may not be in the database and a latent
    semantics file, identifies and visualizes the most similar n images under the selected latent semantics.
    returns a dict like {'image-cc-1-1.png': 0.0,
                        'image-cc-27-6.png': 0.1766516010183068,
                        'image-cc-16-8.png': 0.22128640292896762,
                        'image-cc-12-5.png': 0.24852591178358807,
                        'image-cc-16-10.png': 0.32407461699227724}
    '''
    model=latent_semantics_file['Model']
    cluster_centers=latent_semantics_file['Centroids']
    features = vector

    _weight_pair=latent_semantics_file['Weight_Pair']
    
    #_weight_pair_query = OrderedDict()
    latent_features = {}
    #for each in query_image_name:
    #   _weight_pair_query[each]=[0]*K
    
    # First, map our input image vector to the latent feature space
    mapped_vector = []
    for each in cluster_centers:
        distance=0
        if(model=='color'):
            distance = sum(abs(val1-val2) for val1,val2 in zip(each,features))
        if (model=='elbp' or model=='hog'):
            distance = scipy.spatial.distance.euclidean(each,features)
        mapped_vector.append(distance)
    #print(mapped_vector)

    # Next, create our mapped vectors for ALL images in the database
    for i in range(len(query_image_name)):
        filename=query_image_name[i]
        feature_vec = data_array[i]
        latent_semantics=[]
        for each in cluster_centers:
            distance=0
            if(model=='color'):
               distance = sum(abs(val1-val2) for val1,val2 in zip(each,feature_vec))
            if (model=='elbp' or model=='hog'):
                distance = scipy.spatial.distance.euclidean(each,feature_vec)
            latent_semantics.append(distance)
        latent_features[filename] = latent_semantics
        #_weight_pair_query[k]=[item1+ item2 for (item1,item2) in zip(_weight_pair_query[k],latent_semantics)]
    
    #print(latent_features)
    # Now that we have computed the latent features, we can compute the top N features by comparing our
    # input image vector to all images in the DB
    output_list = [None] * len(query_image_name)
    for i in range(len(query_image_name)):
        output_list[i] = (query_image_name[i], vector_distance(latent_features[query_image_name[i]], mapped_vector))
    return sorted(output_list, key=lambda x: x[1])[:n]
    """
    sum_weight_pair_query={}
    for each in _weight_pair_query.keys():
        sum_weight_pair_query[each]=sum(_weight_pair_query[each])
        
    sum_weight_pair={}
    for each in _weight_pair.keys():
        sum_weight_pair[each]=sum(_weight_pair[each])
        
    weight_difference={}
    for each in sum_weight_pair.keys():
        weight_difference[each]=abs((sum_weight_pair[each])
        -sum_weight_pair_query[query_image_name[0]])
        
    return ((dict((sorted(weight_difference.items(),key = lambda x:x[1]))[:n])))
    """
   
def similarity_classification(vector,latent_semantics_file,data_array,image_dict,K,subject_bool):
    model=latent_semantics_file['Model']
    cluster_centers=latent_semantics_file['Centroids']
    features = vector
    switch = 'subject' if subject_bool else 'type'

    _weight_pair=latent_semantics_file['Weight_Pair']
    
    #_weight_pair_query = OrderedDict()
    temp_dict = {}
    #for each in query_image_name:
    #   _weight_pair_query[each]=[0]*K
    
    # First, map our input image vector to the latent feature space
    mapped_vector = []
    for each in cluster_centers:
        distance=0
        if(model=='color'):
            distance = sum(abs(val1-val2) for val1,val2 in zip(each,features))
        if (model=='elbp' or model=='hog'):
            distance = scipy.spatial.distance.euclidean(each,features)
        mapped_vector.append(distance)
    #print(mapped_vector)

    i = 0
    # Next, create our mapped vectors for ALL images in the database and group by subject/type
    for filename, value_dict in image_dict.items():
        feature_vec = data_array[i]
        latent_semantics=[]
        for each in cluster_centers:
            distance=0
            if(model=='color'):
               distance = sum(abs(val1-val2) for val1,val2 in zip(each,feature_vec))
            if (model=='elbp' or model=='hog'):
                distance = scipy.spatial.distance.euclidean(each,feature_vec)
            latent_semantics.append(distance)
        if value_dict[switch] not in temp_dict:
            temp_dict[value_dict[switch]] = []
        temp_dict[value_dict[switch]].append(vector_distance(latent_semantics, mapped_vector))
        i += 1
       
    # Now, compute the average distances for each subject/type
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
    return np.linalg.norm(np.array(vector_1) - np.array(vector_2))

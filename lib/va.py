##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: va.py
# 
# Description: A file to create the VA File index structure.
##################################################################################

import numpy as np
import math

def index_partition(f,b):
    va=np.zeros((len(f),len(f[0])))
    partition_points=np.zeros((len(f[0]),2**b +1))
    #pp=math.floor(len(f)/2**b)
    a=''
    for i in range(len(f[0])):
        dim_i=[]
        for image in f:
            dim_i.append(image[i])
        sorted_ind=np.array(dim_i).argsort()
        split_list=np.array_split(sorted_ind,2**b)
        #print(split_list)
        partition=0
        partition_points[i][0]=f[sorted_ind[0]][i]
        
        for region in split_list:
            for k in region:
                va[k][i]=partition
            partition+=1
            if(partition<2**b):               
                partition_points[i][partition]=(f[region[len(region)-1]][i]+f[split_list[partition][0]][i])/2
        partition_points[i][partition] = f[split_list[-1][-1]][i]
    for img in va:
        for dim in img:
            a+= np.binary_repr(int(dim), width=b)
          
    return va,partition_points,a,len(a)/8
        
            
def get_dist(v_i, v_q, deg=2):
    ret = 0
    for i in range(0, len(v_i)):
        ret += abs(v_i[i] - v_q[i]) ** deg
    return ret ** (float(deg) ** -1.)


def get_bound(a, p, q):
    ret = 0
    n_features = len(a)
    a_q = np.zeros(n_features)

    for i in range(0, n_features):  # Find approximation of query
        for j in range(0, len(p[0])):
            if (q[i] < p[i][j]):
                a_q[i] = j - 1            
                break
            if(j==len(p[0])-1):
                a_q[i]=j

    for i in range(0, n_features):
        if (a[i] < a_q[i]):
            ret += (q[i] - p[i][int(a[i]) + 1]) ** 2
        elif (a[i] > a_q[i]):
            ret += (p[i][int(a[i])] - q[i]) ** 2
    return ret ** (0.5)


def SortOnDst(dst, nearest_k):
    k = len(nearest_k)
    ret_ans = np.zeros(k)
    ret_dst = np.zeros(k)

    ind = np.argsort(dst)
    for i in range(0, k):
        ret_ans[i] = nearest_k[ind[i]]
        ret_dst[i] = dst[ind[i]]
    return ret_dst, ret_ans


def VA_SSA(imgs_features, va, p, q, k):

    n_samples = len(imgs_features)
    n_dim =len(imgs_features[0])
    
    ans = np.zeros(k, dtype=int)
    dst = np.array([math.inf] * k)
    
    
    overall_img_considered=[]
    #num_buckets_searched=set()
    buckets_searched={}
    for i in range(n_dim):
        buckets_searched[i]=set()
    
    for i in range(0, n_samples):
        l_i = get_bound(va[i], p, q)

        if (l_i < dst[k - 1]):
            d_i = get_dist(imgs_features[i], q)
            overall_img_considered.append(i)
            #string_a = [str(x) for x in va[i]]
            #num_buckets_searched.add("-".join(string_a))
            for r in range(n_dim):
                buckets_searched[r].add(tuple(va[r]))
            
            if(d_i < dst[k - 1]):
                dst[k - 1] = d_i
                ans[k - 1] = i
                dst, ans = SortOnDst(dst, ans)
    b=0
    for i in range(n_dim):
        b+=len(buckets_searched[i])
    return ans, len(overall_img_considered), b
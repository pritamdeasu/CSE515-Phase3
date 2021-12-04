##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: svmBinary.py
# 
# Description: A file containing functions used for computing SVM classification
# on data.
##################################################################################


from math import exp

learning_rate = 0.005
epoch = 600
validations = 1
classification = []

def multiClass(s):
    m = list(set(s))
    m.sort()
    for i in range(len(s)):
        new = [0] * len(m)
        new[m.index(s[i])] = 1
        s[i] = new
    return m

def sigmoid(z):
    return 1.0 / (1.0 + exp(-z))

def WTX(Q, X):
    det = 0.0
    for i in range(len(Q)):
        det += X[i] * Q[i]
    return det

# Function to perform Gradient Descent on the weights/parameters
def gradientDescent(omega, c, x, y, learning_rate):
    oldTheta = omega[c]
    for Q in range(len(omega[c])):
        derivative_sum = 0
        for i in range(len(x)):
            derivative_sum += (sigmoid(WTX(oldTheta, x[i])) - y[i]) * x[i][Q]
        omega[c][Q] -= learning_rate * derivative_sum


def predict(data, omega, labeled_vectors, labels):
    classification = []
    predictions = []
    distance=[]
    count = 1
    for row in data:
        hypothesis = []
        multiclass_ans = [0] * len(omega)
        for c in range(len(omega)):
            z = WTX(row, omega[c])
            
            hypothesis.append(sigmoid(z))
        index = hypothesis.index(max(hypothesis))
        multiclass_ans[index] = 1
        predictions.append(multiclass_ans)
        count += 1

    for i in range(len(predictions)):
        if predictions[i]==[1,0]:
            classification.append(0)
            distance.append(abs(WTX(data[i],omega[0])))
        else:
            classification.append(1)
            distance.append(abs(WTX(data[i],omega[1])))
            
    # Experimental section to ALSO get the distances for the labeled data (so
    # we are able to include original data in the second set of results).
    labeled_distances = []
    for i in range(len(labels)):
        if labels[i]==[1,0]:
            labeled_distances.append(abs(WTX(labeled_vectors[i], omega[0])))
        else:
            labeled_distances.append(abs(WTX(labeled_vectors[i], omega[1])))
    
    return classification,distance,labeled_distances

def fit(X,Y):
    #X=X.tolist()
    #Y=Y.values.tolist()
    
    # Minor changes made to parse data correctly.
    labels=multiClass(Y)
    classes=[]
    for i in range(len(labels)):
        classes.append([row[i] for row in Y])
    omega=[[0] * len(X[0]) for _ in range(len(classes))]
    for i in range(epoch):
        for class_type in range(len(classes)):
                gradientDescent(omega,class_type,X,classes[class_type],learning_rate)
        if (i%(epoch/10)==0):
            print("Processed", i * 100 / epoch, "%")
    return omega, Y




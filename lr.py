"""
File Name : lr.py
Author : Swapnil_Agrawal
Date Created : 01/23/2019
Python Version : 3.6
Detail : Predict labels using logistic regression and SGD over the formatted data from feature.py
Input_format :  python lr.py formatted_train.tsv formatted_valid.tsv formatted_test\
.tsv dict.txt train_out.labels test_out.labels metrics_out.txt 60
"""

import numpy as np 
import sys
import csv

if __name__ == '__main__':

    alpha = 0.1
    epoch = int(sys.argv[8])

    # storing words of dictionary
    dict = {}
    reader = csv.reader(open(sys.argv[4], "r"), delimiter='\t')
    D = np.array(list(reader))


def dot(X, theta):
    #sparse dot product
    # X is a dictionary and theta are the corressponding parameters
    product = 0
    for i, a in X.items():
        product = product + a*theta[i]

    return product


def values(data):

    label = np.zeros(len(data))
    x = []
    # will filter out label and attribute value from data
    for i in range(len(data)):

        # x is a list of dictionary which contains feature vector for each movie review
        x.append({}) 
        x[i][-1] = 1  # bias term
        
        for t in range(1, len(data[i])):  #-1 bcoz of space in the end
            a, b = data[i][t].split(":")
            x[i][int(a)] = int(b)

        # label contains the 0/1 value based upon the review of a movie
        label[i] = int(data[i][0])

    return label, x

# def obj(theta, data, x, label):
#     z = np.zeros(len(data))
#     J = np.zeros(len(data)) 


#     # ojective function calculates negative maximum log likelihood and our aim is to minimize it	
#     for i in range(len(data)):
#         z[i] = dot(x[i], theta)  #theta*X
#         J[i] = J[i] + (-label[i])*z[i] + np.log(1 + np.exp(z[i]))  

#     return J, z


def update(theta, label, x, epoch, data):

    # update theta using SGD and for specified number of epochs

    z = np.zeros(len(data))
    for e in range(epoch):
        for i in range(len(data)):
            z[i] = dot(x[i], theta)  

            C = (label[i] - (np.exp(z[i])/( 1 + np.exp(z[i]))))

            for j, a in x[i].items():
                theta[j] = theta[j] + alpha*a*C

    return theta


def predict(theta, data):

    label, X = values(data)

    y_cap = []
    z = np.zeros(len(data))

    for i in range(len(data)):
    
        z[i] = dot(X[i], theta)  #theta*X
        # calculate p(y=1) and p(y=0), whichever is more is prediction
        p1 = np.exp(z[i])/(1 + np.exp(z[i]))
        p0 = 1/(1 + np.exp(z[i]))

        if(p1>=p0):
            y_cap.append(1)
        else:
            y_cap.append(0)

    return y_cap, label


def error(y_cap, label):
    
    err = 0
    for i in range(len(y_cap)):

        if(y_cap[i] != label[i]):
            err = err + 1

    err = err/len(y_cap)

    return err


def main():
 
    for d in D:
        ## storing in dictionary
        a, b = d[0].split()
        dict[a] = b

    #training data
    reader = csv.reader(open(sys.argv[1], "r"), delimiter='\t') 
    train = np.array(list(reader)) 
    # test data
    reader = csv.reader(open(sys.argv[3], "r"), delimiter='\t') 
    test = np.array(list(reader))  
    
    label, X = values(train)

    theta = {i:0 for i in range(len(dict))} 
    #including bias term
    theta[-1] = 0
    # Updated theta value after specified number of epochs
    theta = update(theta, label, X, epoch, train)
    
    # use calclated theta to predict the train and test label
    yp_train, L1 = predict(theta, train)
    err_train = round(error(yp_train, L1), 6)
    yp_test, L2 = predict(theta, test)
    err_test = round(error(yp_test, L2), 6)

    
    # write train predicted label in 5th file 
    writer1 = open(sys.argv[5], "w")
    for y in yp_train:
        writer1.write(str(y))
        writer1.write("\n")

    # write test predicted label in 6th file 
    writer2 = open(sys.argv[6], "w")
    for y in yp_test:
        writer2.write(str(y))
        writer2.write("\n")

    # write train and test error in 7th file
    writer3 = open(sys.argv[7], "w")
    writer3.write("error(train): " + str(err_train))
    writer3.write("\n")
    writer3.write("error(test): " + str(err_test))


main()
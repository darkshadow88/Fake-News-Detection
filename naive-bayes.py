#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Fake news detection
    The naive bayes model

"""

 
#import all the required libraries

from getEmbeddings import getEmbeddings
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import os
import pickle
from sklearn import metrics

def plot_cmat(yte, ypred):
    """
        Plotting confusion matrix
    
    """
    skplt.metrics.plot_confusion_matrix(yte,ypred)
    plt.show()


def naive_bayes_model():
    """
    In this function the naive bayes classified is built
    
    """
    
    '''
        Read the data from all the .npy file if file exist,
        and if not then call the getEmbeddings function to 
        create the .npy files.
        more about getEmbeddings is in the getEmbeddings.py
        NOTE: .npy stands for numpy array
    '''
    if not os.path.isfile('./xtr.npy') or \
        not os.path.isfile('./xte.npy') or \
        not os.path.isfile('./ytr.npy') or \
        not os.path.isfile('./yte.npy'):
        xtr,xte,ytr,yte = getEmbeddings("datasets/train.csv")
        np.save('./xtr', xtr)
        np.save('./xte', xte)
        np.save('./ytr', ytr)
        np.save('./yte', yte)

    #Load the files to local variables.
    xtr = np.load('./xtr.npy')
    xte = np.load('./xte.npy')
    ytr = np.load('./ytr.npy')
    yte = np.load('./yte.npy')
    
    # Use the built-in Naive Bayes classifier of scikit learn library
    '''
        creating the classiier GaussianNB() 
        fitting the model with xte(xtranning) and ytr(ytranning)
    
    '''
    gnb = GaussianNB()
    gnb.fit(xtr,ytr)
    
    #Saving the models in the naive_bayes.sav file so that we can use pretranied model
    
    model_file = 'naive_bayes_model.sav'
    pickle.dump(gnb,open(model_file,'wb'))
    
    #Prediction the y_pred values for xte(xtest) 
    y_pred = gnb.predict(xte)

    #Printing the accuracy of 
    print("Accuracy = " + format(metrics.accuracy_score(yte,y_pred)*100, '.2f') + "%")

    #Draw the confussion matrix
    plot_cmat(yte, y_pred)

#The main function
if __name__ == '__main__':
    naive_bayes_model()
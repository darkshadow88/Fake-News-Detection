#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Fake news detection
    The SVM model

"""

#import all the required libraries

from getEmbeddings import getEmbeddings
import numpy as np
from sklearn.svm import LinearSVC
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

def svm_model():
    """
        In this function the support vector machine classified is built
    
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

    
    # Use the built-in SVM for classification
    '''
        creating the classiier LinearSVC() 
        fitting the model with xte(xtranning) and ytr(ytranning)
    
    '''
    clf = LinearSVC()
    clf.fit(xtr, ytr)
    
    #Saving the models in the svm_model.sav file so that we can use pretranied model
    model_file = 'svm_model.sav'
    pickle.dump(clf,open(model_file,'wb'))

    #Prediction the y_pred values for xte(xtest)
    y_pred = clf.predict(xte)

    #Plotting the confussion matrix    
    print("Accuracy = " + format(metrics.accuracy_score(yte,y_pred)*100, '.2f') + "%")
    
    # Draw the confusion matrix
    plot_cmat(yte, y_pred)

#The main function
if __name__ == '__main__':
    svm_model()
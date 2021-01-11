#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Fake news detection
    The Doc2Vec pre-processing

"""

import numpy as np
import re
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from nltk.corpus import stopwords


def textClean(text):
    """
        Get rid of the non-letter and non-number characters
        and get rid of stop words.
    
    """
    #Passing the regex expression
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    
    # making all the letters lowercase
    text = text.lower().split()
    
    # get rid of stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    
    return (text)


def cleanup(text):
    """
        Cleaning the  text , get rid of punctuation
    """
    
    
    text = textClean(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def constructLabeledSentences(data):
    """
        Utility function for the Doc2Vec model
        It generates the  comma separated list of words in paragraph/article
        and uses a Label for each article as Text_inedex
        NOTE:index is the index in the dataset
    """
    sentences = []
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences

vector_dimension = 300

def Doc2Vec_model(path):
    """
    Generate Doc2Vec training and testing data
    
    """
    # Reading the data source
    data = pd.read_csv(path)

    # Finding missing values in the dataset
    missing_rows = []
    for i in range(len(data)):
        if data.loc[i, 'text'] != data.loc[i, 'text']:
            missing_rows.append(i)
    data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)

    # Calling the cleenup function for each article
    for i in range(len(data)):
        data.loc[i, 'text'] = cleanup(data.loc[i,'text'])

    # Creating the Label sentences
    x = constructLabeledSentences(data['text'])
    
    # Reading the labels as y variable
    y = data['label'].values

    # Creating the Doc2Vec model
    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7, epochs=10,
                         seed=1)
    text_model.build_vocab(x)
    text_model.train(x, total_examples=text_model.corpus_count, epochs=text_model.iter)
    
    return x, y,text_model

def getEmbeddings(path):
    """
        Creating tranning and testing dataset as numpy arrays for faster processing.
    """
    # Calling the Doc2Vec_model to get the model
    x,y,text_model = Doc2Vec_model(path)    
    
    # Spliting the datset for tranning and testing
    train_size = int(0.8 * len(x))
    test_size = len(x) - train_size

    # Creating Numpy arrays with 0 value
    text_train_arrays = np.zeros((train_size, vector_dimension))
    text_test_arrays = np.zeros((test_size, vector_dimension))
    train_labels = np.zeros(train_size)
    test_labels = np.zeros(test_size)

    # Creating Tranning numpy arrays
    for i in range(train_size):
        text_train_arrays[i] = text_model.docvecs['Text_' + str(i)]
        train_labels[i] = y[i]
    
    # Creating Testing numpy arrays
    j = 0
    for i in range(train_size, train_size + test_size):
        text_test_arrays[j] = text_model.docvecs['Text_' + str(i)]
        test_labels[j] = y[i]
        j = j + 1


    return text_train_arrays, text_test_arrays, train_labels, test_labels



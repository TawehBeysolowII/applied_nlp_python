# -*- coding: utf-8 -*-

#Import the necessary modules 
import os
import pandas as pan
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

trials=100

def load_data():
    c = CountVectorizer(stop_words='english', token_pattern=r'\w+')
    l = LabelEncoder()
    data = pan.read_csv('/Users/tawehbeysolow/Downloads/smsspamcollection/SMSSPamCollection.csv',
                        delimiter='\t', 
                        header=None)
    x = c.fit_transform(data[1]).todense()
    y = l.fit_transform(data[0])
    return x, y 

def train_model(trials=trails):
    x, y = load_data()
    train_end = int(math.floor(len(x)*.67))
    train_x, train_y = x[0:train_end, :], y[0:train_end]
    test_x, test_y = x[train_end:, :], y[train_end:]
    rows = np.random.random_integers(0, len(train_x)-1, len(train_x)-1)

    #Fitting training algorithm 
    s = SVC(kernel='rbf')
    
    for i in range(trials):
        s.fit(train_x, train_y)
        predicted_y_values = s.predict(train_x)
        
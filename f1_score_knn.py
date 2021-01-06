#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 04:38:43 2020

@author: sufiyan
"""

#first import the library
import pandas as pd
# datasert load
dataset = pd.read_csv("Iris.csv")
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
# dataset = pd.read_csv(data, names=names)
# print(dataset.head()) #it prints 20 rows of data
#slicing
X_features_input = dataset.iloc[:, :-1].values #features[rows, columms]
# print(X_features_input)
y_label_output = dataset.iloc[:, 4].values #labels
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features_input, y_label_output, test_size=0.20, random_state=5)
#x_train = 80% of our features data(input)
#x_test = 20% of our features data(input)
#y_train = 80% of our lable data(output)
#y_test = 20 % of pur lable data(output)
#imported the algorithms from library
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
# to train the model you have to use the function of "fit()"
# while traininf we only pass the 80 percent of our data
classifier.fit(X_train, y_train) # X_train = features #y_train= lable
# now we have to take prediction on testing data
y_pred = classifier.predict(X_test) #here we only pass the features

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import f1_score
f1_metric = f1_score(y_test, y_pred, average = "macro")
#average="macro" it calculates the sperate precision and recall of
# each class and than take the average of precision and recall. after it calculate the f1 score
print("F1 Score macro:",f1_metric)
#
from sklearn.metrics import f1_score
f1_metric_micro = f1_score(y_test, y_pred, average = "micro")
print("F1 Score Micro:",f1_metric_micro)
# for accuracy
from sklearn.metrics import accuracy_score
print('Accuracy Score: ', accuracy_score(y_pred, y_test)) #y_pred is the output
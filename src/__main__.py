#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemplo do Algoritmo k-NN para scikit-learn.
@author: diegobertolini
"""
import classifiers as cl
import numpy as np

def main():

        # load data
        print ("Loading data...")
        #tr = np.loadtxt(dataTr) ;
        #ts = np.loadtxt(dataTs) ;
        tr = np.loadtxt('features-response/training-images.txt')
        ts = np.loadtxt('features-response/test-images.txt')
        y_test  = ts[:,7]
        y_train = tr[:,7]
        X_train = tr[:, 1 : 7]
        X_test  = ts[:, 1 : 7]

        #print(cl.KNNClassifier(X_train, y_train, X_test, y_test))
        #print(cl.SVMWithGridSearch(X_train, y_train, X_test, y_test))
        #print(cl.MLP(X_train, y_train, X_test, y_test))
        #print(cl.RandomForestClassifiers(X_train, y_train, X_test, y_test))
        print(cl.decisionTree(X_train, y_train, X_test, y_test))

if __name__ == "__main__":
        main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 16:06:13 2022

@author: alban
"""
import algorithm
import numpy as np
from keras.datasets import mnist

class Naive_bayes(algorithm.Algorithm):
    
    def __init__(self):
        pass
    
    def fit(self, trainSet, labelTrainSet):
        # The data have to be normalized
        # Computing a dictionnary of pre treatment
        self.total_in_training = len(trainSet)
        self.count_dictionnary = {}
        for i in range(10):
            self.count_dictionnary[i] = [0, np.zeros((28,28))]
        for index, matrix in enumerate(trainSet):
            actualClass = labelTrainSet[index]
            self.count_dictionnary[actualClass][0] += 1
            self.count_dictionnary[actualClass][1] = np.add(\
                            self.count_dictionnary[actualClass][1], matrix)
    
    def predict(self, testArray):
        prediction = {}
        for i in range(10):
            prediction[i] = 1
        for actualClass in range(10):
            for index, value in np.ndenumerate(testArray):
                if value == 1:
                    prediction[actualClass] *= (\
                    (self.count_dictionnary[actualClass][1][index] + 1) \
                        / (self.count_dictionnary[actualClass][0] + 10))
                else :
                    prediction[actualClass] *= (\
                    (self.count_dictionnary[actualClass][0] - \
                     self.count_dictionnary[actualClass][1][index] + 1) \
                        / (self.count_dictionnary[actualClass][0] + 10))
            prediction[actualClass] *= self.count_dictionnary[actualClass][0]\
                / self.total_in_training
        #print(prediction)
        return self.take_max(prediction)
            
    def take_max(self, prediction):
        maximum = 0
        return_value = -1
        for key, value in prediction.items():
            if value > maximum:
                maximum = value
                return_value = key
        return return_value
    
    def score(self, testSet, labelTestSet):
        count_good = 0
        dicoPrediction = {}
        for index, test in enumerate(testSet):
            value = self.predict(test)
            print(index, " - Prediction = ", value, \
                  " | Real = ", labelTestSet[index])
            if value == labelTestSet[index]:
                count_good += 1
            dicoPrediction[index] = value
        return dicoPrediction, count_good / len(testSet)
    
if __name__ == "__main__":
    alg = Naive_bayes()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train[0] // 128)
    print(np.shape(X_train))
    alg.fit(X_train// 128, y_train)
    dico, acc = alg.score([X_test[0] //128], [y_test[0]])
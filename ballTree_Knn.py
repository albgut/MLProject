#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 22:40:20 2022

@author: alban
"""
import algorithm
import pickle
from dataclasses import dataclass, field
import numpy as np
import heapq as h
from ball_tree_test import BallTree, item
import distance
import bdd

class BallTree_Knn(algorithm.Algorithm):
    
    def __init__(self, k, distance):
        self.dist = distance
        self.k = k
        self.queue = []
        
        print("load pickle")
        with open('./Data/BallTree.pickle', 'rb') as file:
            self.BT = pickle.load(file)
        print("end loading")
    
    def fit(self, trainSet, labelTrainSet):
        #self.BT = BallTree(trainSet.reshape(len(trainSet), 28*28), labelTrainSet)
        
        pass
        
            
    def predict(self, testArray):
        self.queue = []
        self.queue = self.knn_search(testArray, self.k, self.queue, self.BT)
        npq = np.array([x.label for x in self.queue])
        return np.argmax(np.bincount(npq))
    
    def score(self, testSet, labelTestSet):
        count_good = 0
        dicoPrediction = {}
        for index, test in enumerate(testSet):
            self.queue = []
            value = self.predict(test)
            print(index, " - Prediction = ", value, \
                  " | Real = ", labelTestSet[index])
            if value == labelTestSet[index]:
                count_good += 1
            dicoPrediction[index] = value
        return dicoPrediction, count_good / len(testSet)
    
    def knn_search(self, test, k, queue, BT):
        if len(queue) == k and self.dist.computeDistance(test, BT.loc) - BT.radius >= -queue[0].distance:
            return queue
        elif BT.child1 == None and BT.child2 == None:
            for index, image in enumerate(BT.data):
                if len(queue) == 0 or self.dist.computeDistance(test, image) < -queue[0].distance:
                    h.heappush(queue, item(-self.dist.computeDistance(test, image), BT.labels[index]))
                    if len(queue) > k:
                        h.heappop(queue)
        else:
            d1 = self.dist.computeDistance(test, BT.child1.loc)
            d2 = self.dist.computeDistance(test, BT.child2.loc)
            if min(d1, d2) == d1:
                child1 = BT.child1
                child2 = BT.child2
            else:
                child1 = BT.child2
                child2 = BT.child1
            self.knn_search(test, k, queue, child1)
            self.knn_search(test, k, queue, child2)
        return queue
    
if __name__ == "__main__":
    """
    alg = BallTree_Knn(5, distance.L2())
    testSet, labelTestSet = bdd.getTestImageLabelsFlatten(True)
    alg.score(testSet, labelTestSet)
    """
    """
    alg = BallTree_Knn(5, distance.L2())
    print(alg.BT.loc)
    """
    
    
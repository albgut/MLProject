#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:42:33 2022

@author: alban
"""
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from keras.datasets import mnist
import algorithm
import naive_bayes
import heapq as h
from dataclasses import dataclass, field
from typing import Any
import pickle as p

class BallTree:
    """Simple Ball tree class"""

    def __init__(self, data, labels):
        self.data = np.asarray(data)
        self.labels = np.asarray(labels)

        self.loc = data.mean(0)
        self.radius = np.sqrt(np.max(np.sum((self.data - self.loc) ** 2, 1)))

        self.child1 = None
        self.child2 = None
        
        self.split_point = self.loc

        if len(self.data) > 1:
            largest_dim = np.argmax(self.data.max() - self.data.min())
            i_sort = np.argsort(self.data[:, largest_dim])
            self.data[:] = self.data[i_sort, :]
            self.labels[:] = self.labels[i_sort]

            N = self.data.shape[0]
            half_N = int(N / 2)
            self.split_point = 0.5 * (self.data[half_N, largest_dim]
                                 + self.data[half_N - 1, largest_dim])

            self.child1 = BallTree(self.data[half_N:], self.labels[half_N:])
            self.child2 = BallTree(self.data[:half_N], self.labels[:half_N])            
                
def dist(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

@dataclass(order=True)
class item:
    distance: int
    label: int=field(compare=False)

def k_nn_version_np(test, k, arrayLabels, arrayDist, BT):
    if len(arrayDist) == k and dist(test, BT.loc) - BT.radius >= dist(test, arrayDist[0]):
        #print("finish = ", arrayDist)
        #print(arrayLabels)
        return arrayLabels, arrayDist
    elif BT.child1 == None and BT.child2 == None:
        for index, image in enumerate(BT.data):
            if len(arrayDist) == 0 or dist(test, image) < arrayDist[0]:
                arrayDist = np.append(arrayDist, dist(test, image))
                arrayLabels = np.append(arrayLabels, BT.labels[index])
                i_sort = np.argsort(- arrayDist)
                #print(arrayLabels)
                #print(arrayDist)
                arrayDist[:] = arrayDist[i_sort]
                #arrayImage[:] = arrayImage[i_sort,:]
                arrayLabels[:] = arrayLabels[i_sort]
                if len(arrayDist) > k:
                    #print("before = ", arrayDist)
                    arrayDist = arrayDist[1:]
                    arrayLabels = arrayLabels[1:]
                    #print("after = ", arrayDist)
    else:
        d1 = dist(test, BT.child1.loc)
        d2 = dist(test, BT.child2.loc)
        if min(d1, d2) == d1:
            child1 = BT.child1
            child2 = BT.child2
        else:
            child1 = BT.child2
            child2 = BT.child1
        arrayLabels, arrayDist = k_nn_version_np(test, k, arrayLabels, arrayDist, child1)
        arrayLabels, arrayDist = k_nn_version_np(test, k, arrayLabels, arrayDist, child2)
    return arrayLabels, arrayDist

def printResult(arrayNumbers, arrayTotalNumbers):
    for i in range(len(arrayNumbers)):
        print(i, " ----- ", arrayNumbers[i] / arrayTotalNumbers[i])
                
def knn_search(test, k, queue, BT):
    if len(queue) == k and dist(test, BT.loc) - BT.radius >= -queue[0].distance:
        return queue
    elif BT.child1 == None and BT.child2 == None:
        for index, image in enumerate(BT.data):
            if len(queue) == 0 or dist(test, image) < -queue[0].distance:
                h.heappush(queue, item(-dist(test, image), BT.labels[index]))
                if len(queue) > k:
                    h.heappop(queue)
    else:
        d1 = dist(test, BT.child1.loc)
        d2 = dist(test, BT.child2.loc)
        if min(d1, d2) == d1:
            child1 = BT.child1
            child2 = BT.child2
        else:
            child1 = BT.child2
            child2 = BT.child1
        knn_search(test, k, queue, child1)
        knn_search(test, k, queue, child2)
    return queue

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    alg = naive_bayes.Naive_bayes()
    alg.change_several_in_numpy_01(X_train)
    alg.change_several_in_numpy_01(X_test)
    """
    BT = BallTree(X_train.reshape(60000, 28*28), y_train)
    test = X_test.reshape(10000, 28*28)
    test = X_test[0].reshape(1, 28*28)

    print("finish train")
    with open('./Data/BallTree.pickle', 'wb') as file:
        p.dump(BT, file)
        
        
    
    with open('./Data/BallTree.pickle', 'rb') as file:
        BT = p.load(file)
    """
    
    acc = [0 for i in range(15)]
    """
    indexTest = 1200
    k = 1
    q = []
    q = knn_search(test[indexTest], k, q, BT)
    npq = np.array([x.label for x in q])
    print(npq, " ", y_test[indexTest])
    print(np.argmax(np.bincount(npq)))
    """
    """ MAUVAISES PREDICTIONS
    for index in [24,43,77,115]:
        g = []
        g = knn_search(test[index], 5, g, BT)
        npg = np.array([x.label for x in g])
        print(y_test[index], 'pred =  ', np.argmax(np.bincount(npg)))
        print(X_test[index])
    """
    kValue = [6, 4, 2]
    for k in kValue:
        good = 0
        total = 0
        arrayNumbers = [0 for i in range(10)]
        arrayTotalNumbers = [0 for i in range(10)]
        for indexTest in range(len(test)):
            total += 1
            q = []
            q = knn_search(test[indexTest], k, q, BT)
            npq = np.array([x.label for x in q])
            if y_test[indexTest] == np.argmax(np.bincount(npq)):
                good += 1
                arrayNumbers[y_test[indexTest]] +=1
            arrayTotalNumbers[y_test[indexTest]] += 1
                
            #print("actual accuracy = ", good / total)
                                           
        print("-------------------------------------------------------------------")
        print("||                           K = ",k,"                             ||")
        print("-------------------------------------------------------------------")
        print('acc = ', good / total)
        print()
        printResult(arrayNumbers, arrayTotalNumbers)
        acc[k] =  good / total
    
    """
    for k in range(1,15):
        good = 0
        for indexTest in range(len(test)):
            currentBT = BT
            
            t = test[indexTest]
            child1 = currentBT.child1
            child2 = currentBT.child2
            
            while (child1 != None and child2 != None) and len(currentBT.labels) > k:
                d1 = np.sqrt(np.sum(np.square(t - child1.loc)))
                d2 = np.sqrt(np.sum(np.square(t - child2.loc)))
                if min(d1, d2) == d1:
                    currentBT = child1
                else:
                    currentBT = child2
                child1 = currentBT.child1
                child2 = currentBT.child2
                
            #print(currentBT.data.reshape(28,28))
            #print(currentBT.labels)
            #print(y_test[indexTest])
            if y_test[indexTest] == np.argmax(np.bincount(currentBT.labels)):
                good += 1
        print('acc = ', good / len(X_test))
        acc[k] =  good / len(X_test)
    """
    print(acc)

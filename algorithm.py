#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:53:46 2022

@author: alban
"""
from abc import ABC
from abc import abstractmethod
import numpy as np
from enum import Enum

"""
class AlgosEnum(Enum):
    NAIVE_BAYES = 1
    KNN = 2
    CNN = 3
    
def mapperToAlgos(algoEnum):
    bdd.init()
    if algoEnum.value == 1:
        return naive_bayes.Naive_bayes()
    if algoEnum.value == 2:
        return ball_tree_test.BallTree(data, labels)
    if algoEnum.value == 3:
        return cnn.cnn()
"""

class Algorithm(ABC):
    
    def change_several_in_numpy_01(self, setArray):
        for m in setArray:
            m = np.asarray(m, "uint8")
            m //= 128
        
    @abstractmethod
    def fit(self, trainSet, labelTrainSet):
        pass
    
    @abstractmethod
    def predict(self, testSet):
        pass
    
    @abstractmethod
    def score(self, testSet, labelTestSet):
        pass
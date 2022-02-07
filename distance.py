#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:26:17 2022

@author: alban
"""
from abc import ABC
from abc import abstractmethod
import numpy as np
#import Levenshtein as L
import time

class Distance(ABC):
    
    @abstractmethod
    def computeDistance(self, matrix1, matrix2):
        pass
    
class L2(Distance):
    
    def __init__(self):
        pass
    
    def computeDistance(self, matrix1, matrix2):
        return np.sqrt(np.sum(np.square(matrix1 - matrix2)))
        #return np.linalg.norm(matrix1 - matrix2)
    
class L1(Distance):
    
    def __init__(self):
        pass
    
    def computeDistance(self, matrix1, matrix2):
        return np.linalg.norm(matrix1 - matrix2, 1)
    
class Levenstein(Distance):
    
    def __init__(self):
        pass
    
    def computeDistance(self, matrix1, matrix2):
        pass
        #return L.distance(str(matrix1), str(matrix2))
        """
        MA VERSION LEVENSHTEIN
        matrix1 = matrix1.flatten()
        matrix2 = matrix2.flatten()
        m = len(matrix1) + 1
        n = len(matrix2) + 1
        cost = np.empty((m, n), dtype='int')
        for i in range(m):
            cost[i][0] = i
        for j in range(n):
            cost[0][j] = j
        for i in range(1, m):
            for j in range(1, n):
                dH = cost[i - 1][j] + 1
                dV = cost[i][j - 1] + 1
                if matrix1[i - 1] == matrix2[j - 1]:
                    dD = cost[i - 1][j - 1]
                else:
                    dD = cost[i - 1][j - 1] + 2
                cost[i][j] = min(dD, dH, dV)
        return cost[m - 1][n - 1]
        """
    
if __name__ == "__main__":
    #d1 = np.linalg.norm(np.array([[1,0,1],[0,0,1],[0,1,0]]) - np.array([[0,1,0],[0,0,1],[0,1,0]]))
    #print(d1)
    """
    a = np.array([[1.41, 2],[5, 0], [4, 4], [1,3],[2,1]])
    t = np.argsort(a, axis=0)
    print(t)
    """
    matrix1 = np.array([0,1])
    matrix2 = np.array([1,0])
    print(matrix1.flatten())
    #t = Levenstein()
    #print(str(matrix1))
    t= L2() 
    print(t.computeDistance(matrix1, matrix2))
    print(np.linalg.norm(matrix1 - matrix2, 2))
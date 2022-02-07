#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 17:31:10 2022

@author: alban
"""
from tensorflow import keras
import algorithm
import numpy as np
from tensorflow.keras.layers import *

class cnn(algorithm.Algorithm):
    
    def __init__(self):
        self.file = './Data/cnn_model.h5'
        pass
    
    def fit(self, trainSet, labelTrainSet):
        # Model / data parameters
        num_classes = 10
        input_shape = (28, 28, 1)
        
        # the data, split between train and test sets
        
        
        # Scale images to the [0, 1] range
        #x_train = trainSet.astype("float32") / 255
        
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(trainSet, -1)
             
        
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(labelTrainSet, num_classes)
        
        self.model = keras.Sequential()
        
        self.model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation='relu', padding='Same', input_shape = (28, 28, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation='relu', padding='Same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(strides=(2,2)))
        self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(strides=(2,2)))
        self.model.add(Dropout(0.25))
        

        self.model.add(Flatten())
        self.model.add(Dense(units=1024, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(units=1024, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(units=10, activation='softmax'))
        
        
        batch_size = 128
        epochs = 15
        
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
    
    def predict(self, testSet):
        #x_test = testSet.astype("float32") / 255
        x_test = np.expand_dims(testSet, -1)  
        return np.argmax(self.model.predict(x_test))
    
    def saveModel(self):
        self.model.save(self.file)
        
    def loadModel(self):
        self.model = keras.models.load_model(self.file)
    
    def score(self, testSet, labelTestSet):
        
        count_good = 0
        dicoPrediction = {}
        for index, test in enumerate(testSet):
            value = self.predict(np.array([test]))
            print(index, " - Prediction = ", value, \
                  " | Real = ", labelTestSet[index])
            if value == labelTestSet[index]:
                count_good += 1
            dicoPrediction[index] = value
        return dicoPrediction, count_good / len(testSet)
        """    
        num_classes = 10
        
        #x_test = testSet.astype("float32") / 255
        x_test = np.expand_dims(testSet, -1)  
        
        y_test = keras.utils.to_categorical(labelTestSet, num_classes)
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print(score) 
        """
    
if __name__ == "__main__":
    """
    t = np.array(
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    alg = cnn()
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    #print(x_test[5])
    alg.loadModel()
    #alg.fit(x_train//128, y_train)
    #alg.model.summary()
    print(alg.predict(np.array([x_test[4]])), 'values = ', y_test[4])
    print(alg.predict(np.array([x_test[45]])), 'values = ', y_test[45])
    print(alg.score(x_test // 128, y_test)[1])
    #alg.model.save('./Data/cnn_model.h5')


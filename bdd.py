#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 16:03:05 2022

@author: alban
"""
import sqlite3

from sqlite3 import Error
from keras.datasets import mnist
import numpy as np
import datetime
import naive_bayes
import time
import ballTree_Knn
import cnn
import distance
from ball_tree_test import BallTree, item

#mode journalisation ?
def create_connection(path):

    connection = None

    try:

        connection = sqlite3.connect(path, timeout=10)

        print("Connection to SQLite DB successful")

    except Error as e:

        print(f"The error '{e}' occurred")


    return connection

def printExp():
    con = create_connection('./Data/digits.db')
    r = con.execute("select * from exp")
    for i in r:
        print(i)
    con.close()
    
def delete():
    con = create_connection('./Data/digits.db')
    con.execute("delete from predictions")
    con.commit()
    con.close()

def getTrainImageLabels(boolean01):
    if boolean01:
        x = 128
    else:
        x = 1
    connect = create_connection('./Data/digits.db')
    row = connect.execute("select image from train").fetchall()
    trainSet = np.zeros((len(row), 28, 28), dtype=np.uint8)
    for index, r in enumerate(row):
        trainSet[index] = (np.frombuffer(r[0], dtype=np.uint8).reshape(28,28) // x)
    row = connect.execute("select label from train").fetchall()
    labelSet = np.zeros((len(row)), dtype=np.uint8)
    for index, r in enumerate(row):
        labelSet[index] = np.frombuffer(r[0], dtype=np.uint8)
    connect.close()
    return trainSet, labelSet

def getTestImageLabels(boolean01):
    if boolean01:
        x = 128
    else:
        x = 1
    connect = create_connection('./Data/digits.db')
    row = connect.execute("select image from test").fetchall()
    testSet = np.zeros((len(row), 28, 28), dtype=np.uint8)
    for index, r in enumerate(row):
        testSet[index] = (np.frombuffer(r[0], dtype=np.uint8).reshape(28,28) // x)
    row = connect.execute("select label from test").fetchall()
    labelSet = np.zeros((len(row)), dtype=np.uint8)
    for index, r in enumerate(row):
        labelSet[index] = np.frombuffer(r[0], dtype=np.uint8)
    connect.close()
    return testSet, labelSet

def getTestImageLabelsFlatten(boolean01):
    if boolean01:
        x = 128
    else:
        x = 1
    connect = create_connection('./Data/digits.db')
    row = connect.execute("select image from test").fetchall()
    testSet = np.zeros((len(row), 784), dtype=np.uint8)
    for index, r in enumerate(row):
        testSet[index] = (np.frombuffer(r[0], dtype=np.uint8).reshape(784) // x)
    row = connect.execute("select label from test").fetchall()
    labelSet = np.zeros((len(row)), dtype=np.uint8)
    for index, r in enumerate(row):
        labelSet[index] = np.frombuffer(r[0], dtype=np.uint8)
    connect.close()
    return testSet, labelSet

def getTotalDigitFromTest(digit):
    con = create_connection('./Data/digits.db')
    numberDigit = con.execute("select count(*) from test where test.label = ? ", (digit.to_bytes(1, 'big'),)).fetchone()
    con.close()
    return numberDigit[0]

def getDigitStatFromExp(idExp, digit):
    con = create_connection('./Data/digits.db')
    listDigit = con.execute("select * from predictions where predictions.idExp = ? and predictions.prediction = ?", (idExp, digit)).fetchall()
    totalDigit = len(listDigit)
    numberCorrect = 0
    for prediction in listDigit:
        trueResult = con.execute("select label from test where test.id = ?", (prediction[1],)).fetchone()
        #trueResult = con.execute("select label from test where test.id = ?", (prediction[1] + 1,)).fetchone()        
        #print(int.from_bytes(trueResult[0], 'big'), " ---- ", digit)
        if int.from_bytes(trueResult[0], 'big') == digit:
            numberCorrect += 1
    con.close()
    numberTotalInTest = getTotalDigitFromTest(digit)
    precision = (numberCorrect / totalDigit)
    sensibilite = (numberCorrect / numberTotalInTest)
    f_mesure = 2 * ((precision * sensibilite) / (precision + sensibilite))
    return precision, sensibilite, f_mesure
    
def stringAlgoToInt(name):
    con = create_connection('./Data/digits.db')
    algoInt = con.execute("select * from algo where algo.name = ?", (name,)).fetchone()
    con.close()
    return algoInt[0][0]

def getExpsId():
    con = create_connection('./Data/digits.db')
    rawList = con.execute("select * from exp").fetchall()
    con.close()
    listExp = []
    for element in rawList:
        listExp.append(element[0])
    return listExp

def getExp(idExp):
    con = create_connection('./Data/digits.db')
    exp = con.execute("select * from exp where id = ?", (idExp,)).fetchone()
    con.close()
    return exp

def getExpName(idExp):
    con = create_connection('./Data/digits.db')
    exp = con.execute("select * from exp where id = ?", (idExp,)).fetchone()
    algo = con.execute("select name from algo where id = ?", (exp[2],)).fetchone()
    name = str(idExp) + " - " + algo[0]
    if algo[0] == 'KNN':
        name += " K = " + str(exp[5]) + ", dist = " + exp[6]
    con.close()
    return name

#    connection.execute("create table exp (id integer primary key, date DATETIME, idAlgo integer, timeExec integer, accuracy float, k int, distance text, FOREIGN KEY(idAlgo) REFERENCES algo(id))")
def add_exp(date, algo, timeExec, accuracy, k, distance, predictionDico):
    print("Saving experiment in database")
    # PATH EN DUR ICI !
    connection = create_connection('./Data/digits.db')
    idAlgo = connection.execute("select id from algo where algo.name = ?", (algo,)).fetchone()[0]
    connection.execute("insert into exp(date, idAlgo, timeExec, accuracy, k, distance) values (?,?,?,?,?,?)", \
                       (date, idAlgo, timeExec, accuracy, k, distance))
    idExp = connection.execute("select id from exp where exp.date = ?", (str(date),)).fetchone()[0]
    for index in predictionDico.keys():
        connection.execute("insert into predictions(idExp, idTest, prediction) values (?,?,?)",\
                           (idExp, index + 1, str(predictionDico[index])))
    connection.commit()
    connection.close()

#path = './Data/digits.db'
def init(path):
    connection = create_connection(path)
    
    #delete table if their exists
    connection.execute("drop table if exists train")
    connection.execute("drop table if exists test")
    connection.execute("drop table if exists exp")
    connection.execute("drop table if exists algo")
    connection.execute("drop table if exists predictions")

    #create tables 
    connection.execute("create table train (id integer primary key, image blob, label text)")
    connection.execute("create table test (id integer primary key, image blob, label text)")
    connection.execute("create table exp (id integer primary key, date DATETIME, idAlgo integer, timeExec integer, accuracy float, k int, distance text, FOREIGN KEY(idAlgo) REFERENCES algo(id))")
    connection.execute("create table algo (id integer primary key, name text)")
    connection.execute("create table predictions (idExp integer, idTest int, prediction text, primary key (idExp, idTest))")
    
    #insert the different algorithms
    connection.execute("insert into algo(name) values ('NaiveBayes')")
    connection.execute("insert into algo(name) values ('KNN')")
    connection.execute("insert into algo(name) values ('CNN')")

    #load data from mnist and put it in good form
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.array(X_train)
    X_train = X_train.reshape((len(X_train), 28*28))
    X_test = np.array(X_test)
    X_test = X_test.reshape((len(X_test), 28*28))
    
    #fill the train table
    for index,image in enumerate(X_train):
        connection.execute("insert into train(image, label) values (?,?)", \
                           (image, y_train[index]))
    #fill the test table
    for index,image in enumerate(X_test):
        connection.execute("insert into test(image, label) values (?,?)", \
                           (image, y_test[index]))

    connection.commit()
    connection.close()
    
    
def executeAlgoNB():
    alg = naive_bayes.Naive_bayes()
    trainSet, labelTrainSet = getTrainImageLabels(True)
    testSet, labelTestSet = getTestImageLabels(True)
    date = datetime.datetime.now()
    start = time.perf_counter()
    alg.fit(trainSet, labelTrainSet)
    prediction, acc = alg.score(testSet, labelTestSet)
    end = time.perf_counter()
    #date, algo, timeExec, accuracy, k, distance, predictionDico)
    return (date, 'NaiveBayes', int(end - start), acc, None, None, prediction)

def executeAlgoKNN():
    alg = ballTree_Knn.BallTree_Knn(5, distance.L2())
    trainSet, labelTrainSet = getTrainImageLabels(True)
    testSet, labelTestSet = getTestImageLabelsFlatten(True)
    date = datetime.datetime.now()
    start = time.perf_counter()
    alg.fit(trainSet, labelTrainSet)
    prediction, acc = alg.score(testSet, labelTestSet)
    end = time.perf_counter()
    #date, algo, timeExec, accuracy, k, distance, predictionDico)
    return (date, 'KNN', int(end - start), acc, 5, 'L2', prediction)

def executeAlgoCNN():
    alg = cnn.cnn()
    trainSet, labelTrainSet = getTrainImageLabels(True)
    testSet, labelTestSet = getTestImageLabels(True)
    date = datetime.datetime.now()
    start = time.perf_counter()
    alg.loadModel()
    prediction, acc = alg.score(testSet, labelTestSet)
    end = time.perf_counter()
    return (date, 'CNN', int(end - start), acc, None, None, prediction)

if __name__ == "__main__":
    """
    d = datetime.datetime.now()
    init('./Data/digits.db')
    predictionDico = {0:1,1:2,2:4}
    print(d)
    add_exp(d, 'KNN', 1285, 0.95, None, None, predictionDico)
    """
    #date, algo, timeExec, accuracy, k, distance, predictionDico = executeAlgoNB()
    #add_exp(date, algo, timeExec, accuracy, k, distance, predictionDico)
    
    #date, algo, timeExec, accuracy, k, distance, predictionDico = executeAlgoKNN()
    #add_exp(date, algo, timeExec, accuracy, k, distance, predictionDico)
    
    
    
    #date, algo, timeExec, accuracy, k, distance, predictionDico = executeAlgoCNN()
    #add_exp(date, algo, timeExec, accuracy, k, distance, predictionDico)
    """
    con = create_connection('./Data/digits.db')
    r = con.execute("select * from predictions").fetchall()
    for i in r:
        print(i)
    con.commit()
    con.close()
    """
    
    con = create_connection('./Data/digits.db')
    rawList = con.execute("select * from exp").fetchall()
    con.close()
    for r in rawList:
        print(r)
    
    #delete()
    #getTrainImageLabels(True)
    #for i in range(10):
    #    print(getDigitStatFromExp(1, i))
    #print(getTotalDigitFromTest(0))
    
    #printExp()
    """
    cursor = create_connection('./Data/digits.db')
    r = cursor.execute("select * from predictions").fetchall()
    for row in r:
        print(row)
        
    r = cursor.execute("select * from exp").fetchall()
    for row in r:
        print(row)
      """ 
    #init('./Data/digits.db')
    
    """SEE THE OPERATIONS
    row = (connection.execute("select * from algo"))
    for r in row:
        print(r)
    row = (connection.execute("select image from test"))
    for r in row:
        print(np.frombuffer(r[0], dtype=np.uint8).reshape(28,28))
    """
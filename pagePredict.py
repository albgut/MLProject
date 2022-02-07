#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 07:39:18 2022

@author: alban
"""
from tkinter import *
from tkinter import ttk
import numpy as np
import naive_bayes
import bdd
import cnn
import distance
import ballTree_Knn

fontTimes=('Times',12,'normal')
bigFont = ("Times", 22, 'bold')

class PagePredict(Frame):
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        factor = 5
        canvas_width = 28 * factor
        canvas_height = 28 * factor
        self.border = Frame(self, padx = 10, pady=10)
        self.chooseAlgWidget = ChooseAlgWidget(self.border, highlightthickness=0)
        self.chooseAlgWidget.pack()
        self.border.pack(side=TOP)
        self.drawWidget = Draw_widget(self, factor, width=canvas_width, height=canvas_height,\
                                      borderwidth=1, relief='ridge')
        self.drawWidget.pack(side=LEFT, padx=10, pady=10)
        
        self.pickleCharged = ballTree_Knn.BallTree_Knn(5, distance.L2())
        
        self.buttonFrame = Frame(self)
        self.printResult = PrintResult(self)
        self.printResult.pack(side=RIGHT, padx=10, pady=10)
        self.buttonPredict = Button(self.buttonFrame, font=fontTimes, text="PREDICT", command=self.predictCommand)
        self.buttonPredict.pack(padx=10, pady=10)
        self.buttonClear = Button(self.buttonFrame, font=fontTimes, text="CLEAR", command=self.drawWidget.clear)
        self.buttonClear.pack(padx=10, pady=10)
        self.buttonFrame.pack(side=RIGHT)
        #side=RIGHT, fill=BOTH,
        self.prediction = ""
        
    def predictCommand(self):
        self.selectAlgo()
        if self.algo == None:
            return
        train, label = bdd.getTrainImageLabels(True)
        print(self.drawWidget.arrayImage)
        if isinstance(self.algo, cnn.cnn):
            self.algo.loadModel()
        else:
            self.algo.fit(train, label)
        test = self.drawWidget.arrayImage
        if isinstance(self.algo, cnn.cnn):
            test = np.array([self.drawWidget.arrayImage])
        if isinstance(self.algo, ballTree_Knn.BallTree_Knn):
            test = self.drawWidget.arrayImage.reshape((1, 784))
        self.prediction = self.algo.predict(test)
        print(self.prediction)
        self.printResult.changePrediction(self.prediction)
        
    def selectAlgo(self):
        nameAlgo = self.chooseAlgWidget.selectedValue
        if nameAlgo == "NAIVE BAYES":
            self.algo = naive_bayes.Naive_bayes()
            return
        if nameAlgo == "KNN":
            self.algo = self.pickleCharged
            return
        if nameAlgo == "CNN":
            self.algo = cnn.cnn()
            return
        else:
            self.algo = None
            
class ChooseAlgWidget(Frame):
    
    def __init__(self, parent, **kwargs):
        self.selectedValue = ""
        super().__init__(parent, borderwidth=1, relief='ridge', **kwargs)
        self.text = Label(self, font=fontTimes, text= "SELECT THE ALGORITHM", bd=50)
        self.text.pack(side=LEFT)
        values = ["NAIVE BAYES","KNN","CNN"]
        self.border = Frame(self, bd=10)
        self.comboBox = ttk.Combobox(self.border, values=values, font=fontTimes, state='readonly')
        self.comboBox.pack()
        self.comboBox.bind('<<ComboboxSelected>>', self.registerValue)
        self.border.pack(side=LEFT)
    
    def registerValue(self, event):
        self.selectedValue = self.comboBox.get()
        self.comboBox.selection_clear()
        
class PrintResult(Frame):
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, borderwidth=1, relief='ridge', **kwargs)
        self.textResult = Label(self, font=fontTimes, text= "PREDICTION", bd=50, )
        self.textResult.pack()
        self.predictionText = StringVar()
        self.predictionText.set("")
        self.widgetLabel = Label(self, font=bigFont, textvariable=self.predictionText, bd=50)
        self.widgetLabel.pack()
        
    def changePrediction(self, prediction):
        self.predictionText.set(prediction)

class Draw_widget(Canvas):
    
    def __init__(self, parent, factor, **kwargs):
        super().__init__(parent, **kwargs)
        self.factor = factor
        self.bind("<B1-Motion>", self.draw)
        self.initArray()
        
    def initArray(self):
        self.arrayImage = np.zeros((28,28), dtype=np.uint8)
    
    def draw(self, event):
        black = "#000000"
        x1, y1 = ( event.x - 1 * (self.factor/2) ), ( event.y - 1 * (self.factor/2) )
        x2, y2 = ( event.x + 1 * (self.factor/2) ), ( event.y + 1 * (self.factor/2) )
        self.create_oval( x1, y1, x2, y2, fill = black )
        x = event.x // self.factor
        y = event.y // self.factor
        self.fillNext(y,x)
            
    #Pour grossir les traits
    def fillNext(self, x, y):
        if x < 28 and y < 28 and x >= 0 and y >= 0:
            self.arrayImage[x][y] = 1
            if x - 1 >= 0:
                self.arrayImage[x-1][y] = 1
                if y - 1 >= 0:
                    self.arrayImage[x-1][y-1] = 1
                #if y + 1 < 28:
                #    self.arrayImage[x-1][y+1] = 1
       
    def clear(self):
        self.delete('all')
        self.initArray()
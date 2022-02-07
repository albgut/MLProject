#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 07:39:37 2022

@author: alban
"""

from tkinter import *
from tkinter import ttk
import bdd
from pagePredict import *

class PageExperiments(Frame):
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.border = Frame(self, padx = 10, pady=10)
        self.chooseExpWidget = ChooseExpWidget(self.border, self.updateResult, highlightthickness=0)
        self.chooseExpWidget.pack()
        #self.buttonResult = ButtonResult(self.border, self.value, padx = 10, pady = 10)
        self.border.pack()
        self.result = self.chooseExpWidget.selectedValue
        
        self.accuracy = StringVar()
        self.accuracy.set("ACCURACY    0")
        
        self.borderIn = Frame(self, bd=10, padx=10, pady=10)
        Label(self.borderIn, textvariable=self.accuracy, padx=10, pady=10).pack()
        self.borderIn.pack()
        
        self.border = Frame(self, padx=10, pady=10, borderwidth=1, relief='ridge')
        self.title = StringVar()
        self.title.set("DIGIT - PRECISION   RECALL   F-MESURE")
        Label(self.border, textvariable=self.title, padx=10, pady=10).pack()
        #for i in range(10):
        #    LineStatDigit(self.border, self.result, i, padx=10, pady=10).pack()
        
        self.lbl0 = LineStatDigit(self.border, self.result, 0, padx=10, pady=10)
        self.lbl1 = LineStatDigit(self.border, self.result, 1, padx=10, pady=10)
        self.lbl2 = LineStatDigit(self.border, self.result, 2, padx=10, pady=10)
        self.lbl3 = LineStatDigit(self.border, self.result, 3, padx=10, pady=10)
        self.lbl4 = LineStatDigit(self.border, self.result, 4, padx=10, pady=10)
        self.lbl5 = LineStatDigit(self.border, self.result, 5, padx=10, pady=10)
        self.lbl6 = LineStatDigit(self.border, self.result, 6, padx=10, pady=10)
        self.lbl7 = LineStatDigit(self.border, self.result, 7, padx=10, pady=10)
        self.lbl8 = LineStatDigit(self.border, self.result, 8, padx=10, pady=10)
        self.lbl9 = LineStatDigit(self.border, self.result, 9, padx=10, pady=10)
        
        self.lbl0.pack()
        self.lbl1.pack()
        self.lbl2.pack()
        self.lbl3.pack()
        self.lbl4.pack()
        self.lbl5.pack()
        self.lbl6.pack()
        self.lbl7.pack()
        self.lbl8.pack()
        self.lbl9.pack()
        
        self.border.pack()
        
    def updateResult(self, result):
        self.result = result
        
        self.lbl0.changeValue(self.result, 0)
        self.lbl1.changeValue(self.result, 1)
        self.lbl2.changeValue(self.result, 2)
        self.lbl3.changeValue(self.result, 3)
        self.lbl4.changeValue(self.result, 4)
        self.lbl5.changeValue(self.result, 5)
        self.lbl6.changeValue(self.result, 6)
        self.lbl7.changeValue(self.result, 7)
        self.lbl8.changeValue(self.result, 8)
        self.lbl9.changeValue(self.result, 9)
        
        self.accuracy.set("ACCURACY    " + str(round(bdd.getExp(result)[4], 4)))
        
        
class ButtonResult(Button):  
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.button = Button(parent, text="RESULT", command=self.changeLabel)
        
        
        
class LineStatDigit(Frame):

    def __init__(self, parent, exp, digit, **kwargs):
        super().__init__(parent, **kwargs)
        self.result = StringVar()
        self.changeValue(exp, digit)
        Label(self, textvariable=self.result).pack()
        
    def changeValue(self, exp, digit):
        if exp == -1:
            precision, sensibilite, f_mesure = 0, 0, 0
        else:
            precision, sensibilite, f_mesure = bdd.getDigitStatFromExp(exp, digit)
        self.result.set(str(digit) + "  -  " + str(round(precision, 4)) + "     " + str(round(sensibilite, 4)) + "     " + str(round(f_mesure, 4)))
        
class ChooseExpWidget(Frame):
    
    def __init__(self, parent, command, **kwargs):
        self.command = command
        self.selectedValue = -1
        self.dicoIdName = {}
        super().__init__(parent, borderwidth=1, relief='ridge', **kwargs)
        self.text = Label(self, font=fontTimes, text= "SELECT THE EXPERIENCE", bd=50)
        self.text.pack(side=LEFT)
        idExps = bdd.getExpsId()
        self.buildDico(idExps)
        values = [i for i in self.dicoIdName.keys()]
        self.border = Frame(self, bd=10)
        self.comboBox = ttk.Combobox(self.border, values=values, font=fontTimes, state='readonly')
        self.comboBox.pack()
        self.comboBox.bind('<<ComboboxSelected>>', self.registerValue)
        self.border.pack(side=LEFT)
        
    def buildDico(self, idExps):
        for i in idExps:
            self.dicoIdName[bdd.getExpName(i)] = i

    def registerValue(self, event):
        self.selectedValue = self.dicoIdName[self.comboBox.get()]
        self.comboBox.selection_clear()
        self.command(self.selectedValue)
        
if __name__ == "__main__":
    root = Tk()
    p= PageExperiments(root)
    p.pack()
    root.mainloop()
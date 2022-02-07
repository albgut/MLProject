#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:03:39 2022

@author: alban
"""
from tkinter import *
from tkinter import ttk
from pagePredict import *
from pageExperiments import *
from ball_tree_test import BallTree, item

fontTimes=('Times',12,'normal')
bigFont = ("Times", 22, 'bold')

class Onglet(ttk.Notebook):
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.pack()
        self.pagePredict = PagePredict(self)
        self.pagePredict.pack()
        self.pageExperiments = PageExperiments(self)
        self.pageExperiments.pack()
        self.add(self.pagePredict, text="PREDICTION")
        self.add(self.pageExperiments, text="EXPERIMENTS")


if __name__ == "__main__":
    root = Tk()
    o = Onglet(root)
    o.pack()
    root.mainloop()
    
"""
global array

factor = 10

canvas_width = 28 * factor
canvas_height = 28 * factor


def initArray():
    return [[0 for i in range(28)] for j in range(28)]

array = initArray()

def paint( event ):
   python_green = "#476042"
   black = "#000000"
   x1, y1 = ( event.x - 1 * (factor/2) ), ( event.y - 1 * (factor/2) )
   x2, y2 = ( event.x + 1 * (factor/2) ), ( event.y + 1 * (factor/2) )
   w.create_oval( x1, y1, x2, y2, fill = black )
   array[event.x//factor][event.y//factor] = 1
   
def refresh():
    w.delete('all')
    for line in array:
        print(line)
    array = initArray()

master = Tk()
master.title( "Painting using Ovals" )
w = Canvas(master, 
           width=canvas_width, 
           height=canvas_height)
w.pack(expand = YES, fill = BOTH)
w.bind( "<B1-Motion>", paint )

#message = Label( master, text = "Press and Drag the mouse to draw" )
#message.pack( side = BOTTOM )

b = Button(master, text="OK", command=refresh)
b.pack()
    
mainloop()
"""
import DL
from turtle import *
from turtle import Turtle, Screen
import turtle
from tkinter import *
from svg_turtle import SvgTurtle
from PIL import Image 
from PIL import EpsImagePlugin
import time
from PIL import Image, ImageOps
import matplotlib
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt

EpsImagePlugin.gs_windows_binary = r'E:\Program Files\gs\gs10.00.0\bin\gswin64c'
class num:
    i = 300

def drawButton(x,y,clr, text=None, width=None, height=None, offsideTxt=0):
    temp = Turtle('turtle')
    temp.hideturtle()
    temp.speed('fastest')
    temp.pensize(3)
    temp.pencolor("black")
    temp.penup()
    temp.setposition(x,y)
    temp.pendown()
    temp.fillcolor(clr) 
    width = buttonWidth if width == None else width
    height = buttonHeight if height == None else height
    temp.begin_fill()
    for i in range(2):
        temp.forward(width)
        temp.left(90)
        temp.forward(height)
        temp.left(90)
    temp.end_fill()
    if text != None:
        if offsideTxt != 0:
            temp.penup()
            temp.setposition(x+offsideTxt,y)
            temp.pendown()

        temp.write(text,font=("Courier",30,"normal"))

    temp.penup()
    temp.hideturtle()
    buttons.append([x,y,clr,width,height,text])


def saveScreen(fileName,letter):
    screen.tracer(False)
    screen.tracer(True)
    canvas = screen.getcanvas()
    tur.hideturtle()
    image_path = r'E:\Program Files\vscode\school project\SAVEDIMAGES\LETTERS'

    if not os.path.exists(f"{image_path}\{letter}"):
        os.mkdir(f"{image_path}\{letter}")

    canvas.postscript(file= f"{image_path}\TRASH\{fileName}.eps", width=WIDTH, height=HEIGHT)

    img = Image.open(f"{image_path}\TRASH\{fileName}.eps") 

    img.save(f"{image_path}\TRASH\{fileName}.jpg")  
    newImg = img.crop((95,0,452,336))
    
    num_px = 28 #64
    image28 = newImg.resize((num_px, num_px), Image.Resampling.LANCZOS)
    newImg.save(f"{image_path}\{letter}\{fileName}.jpg")
    num.i += 1
    return

def DrawButtons():
    drawButton(-250,-240, "black")
    drawButton(-250,-180, "white")
    drawButton(-250,-120, "purple")
    drawButton(-250,-60, "blue")
    drawButton(-250,0, "green")
    drawButton(-250,60, "yellow")
    drawButton(-250,120, "orange")
    drawButton(-250,180, "red")

    drawButton(-110,-270, "#4d2d1c", "SAVE", buttonWidth*2, buttonHeight)
    drawButton(90,-270, "#4d2d1c", "DELETE", buttonWidth*2, buttonHeight)
    drawButton(-110,-210, "#efbc9b", "SPACE", 200 + buttonWidth*2, buttonHeight, offsideTxt = 120)
    tur.color("black")

def ButtonClick(x,y):
    for button in buttons:
        if x > button[0] and x < button[0] + button[3]:
            if y > button[1] and y < button[1] + button[4]:
                if button[5] == None:
                    tur.color(button[2])
                if button[5] == "SAVE":
                    SaveButtonClick(LETTER)
                if button[5] == "DELETE":
                    ClearScreen()
                
                if button[5] == "SPACE":
                    pass

def SaveButtonClick(letter):
    saveScreen(num.i ,letter)
    ClearScreen()

def ClearScreen():
    tur.ht()
    tur.clear()
    tur.st()

def on_motion(event):
    global mouse_x, mouse_y
    mouse_x = event.x - turtle.window_width() / 2
    mouse_y = -event.y + turtle.window_height() / 2

def tick():
    print("TICKK")

    tur.setposition(mouse_x, mouse_y)
    listen()
    tur.onclick(tur.ondrag(drag))
    tur.onrelease(RaisePen)

def drag(i, j):
    tur.pendown()
    tur.ondrag(None)   
    tur.setheading(tur.towards(i, j))
    tur.goto(i, j)
    tur.onrelease(RaisePen)
    tur.ondrag(drag)

def RaisePen(x,y):
    while True:
        tur.penup()
        tur.setposition(mouse_x,mouse_y)
        tur.onclick(tur.ondrag(drag))

    screen.ontimer(RaisePen(0,0), frame_delay_ms)

def main():
    tur.showturtle()
    while True:
        tur.goto(mouse_x,mouse_y)
        tur.onclick(tur.ondrag(drag))
        tur.onrelease(RaisePen)
        onscreenclick(ButtonClick, 1)

    screen.mainloop()

LETTER ="bg"
turtle.title("My Turtle Program")
WIDTH, HEIGHT = 600, 600
screen = turtle.Screen()
screen.setup(WIDTH, HEIGHT)
turtle.screensize(canvwidth=WIDTH, canvheight=HEIGHT,bg="grey")
buttons = []
buttonHeight = 50
buttonWidth = 80
frame_delay_ms = 1000 //100
ws = turtle.getcanvas()
turtle.getcanvas().bind("<Motion>", on_motion)
tur = Turtle('turtle')
tur.ht()
tur.speed(0)
tur.pensize(10)
tur.pencolor("black")
tur.shape("circle")
tur.turtlesize(1,1)

tur.hideturtle()
tur.penup()
mouse_x, mouse_y = 0,0
DrawButtons()



if __name__ == main():
    tur.penup()
    main()


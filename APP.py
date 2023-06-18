from DL import * 
from turtle import *
from turtle import Turtle
import turtle
from tkinter import *
from PIL import Image 
from PIL import EpsImagePlugin
from PIL import Image, ImageOps
import matplotlib
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt

#DOWNLOAD GS FROM https://ghostscript.com/releases/gsdnld.html
#INSERT GS PATH IN THE FOLLOWING LINE
EpsImagePlugin.gs_windows_binary = r'E:\Program Files\gs\gs10.01.1\bin\gswin64c'

class num:
    i = 0
class string:
    st = ""
class lastTur:
    lastTurtle = Turtle('turtle')
    lastTurtle.ht()
    Flag = True

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
    image_path = r'SAVEDIMAGES\LETTERS'

    if not os.path.exists(f"{image_path}\{letter}"):
        os.mkdir(f"{image_path}\{letter}")

    canvas.postscript(file= f"{image_path}\TRASH\{fileName}.eps", width=WIDTH, height=HEIGHT)
    img = Image.open(f"{image_path}\TRASH\{fileName}.eps") 
    img.save(f"{image_path}\TRASH\{fileName}.jpg")  
    newImg = img.crop((95,0,452,336))
    newImg.save(f"{image_path}\{letter}\{fileName}.jpg")
    
def DrawButtons():
    drawButton(-250,-240, "black")
    drawButton(-250,-180, "white")
    drawButton(-250,-120, "purple")
    drawButton(-250,-60, "blue")
    drawButton(-250,0, "green")
    drawButton(-250,60, "yellow")
    drawButton(-250,120, "orange")
    drawButton(-250,180, "red")

    drawButton(-110,-270, "#4d2d1c", "SAVE", buttonWidth*1.25, buttonHeight)
    drawButton(10,-270, "#4d2d1c", "CLEAR", buttonWidth*1.25 + 20, buttonHeight)
    drawButton(150,-270, "#4d2d1c", "UNDO", buttonWidth*1.25, buttonHeight)

    drawButton(-110,-210, "#efbc9b", "SPACE", 200 + buttonWidth*2, buttonHeight, offsideTxt = 120)
    tur.color("black")

def ButtonClick(x,y):
    for button in buttons:
        if x > button[0] and x < button[0] + button[3]:
            if y > button[1] and y < button[1] + button[4]:
                if button[5] == None:
                    tur.color(button[2])
                if button[5] == "SAVE":
                    SaveButtonClick('CURRENTRUN')
                if button[5] == "CLEAR":
                    ClearScreen()
                if button[5] == "UNDO":
                    drawTextOnCanvas(True)                
                if button[5] == "SPACE":
                    updateSTR(char=' ')

def SaveButtonClick(letter):
    saveScreen(num.i ,letter)
    ClearScreen()
    updateSTR(path=r'SAVEDIMAGES\LETTERS\CURRENTRUN\{number}.jpg'.format(number=num.i))
    num.i += 1

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

def loadData():
    dataSizePer = 288
    dataSet = {}
    X, Y = [],[]
    num_px = 28
    letters = {1:'א', 2:"ב",3:"ג",4:"ד",5:"ה",6:"ו",7:"ז",8:"ח",9:"ט",10:"י",11:"כ",12:"ל",13:"מ",14:"נ",15:"ס",16:"ע",17:"פ",18:"צ",19:"ק",20:"ר",21:"ש",22:"ת"}
    for i in range(1,len(letters)+1): 
        target = i 
        for j in range(dataSizePer):
            img_path = r"SAVEDIMAGES\LETTERS\{lete}\{idx}.jpg".format(lete=letters[i], idx=j)
            image = Image.open(img_path)
            image28 = image.resize((num_px, num_px), Image.Resampling.LANCZOS)
            gray_image = ImageOps.grayscale(image28)
            my_image = np.reshape(gray_image,(num_px*num_px, ))
            
            X.append(asarray(my_image)/255 - 0.5)
            Y.append(i-1)


    X = np.array(X)
    Y = np.array(Y)

    p = np.random.permutation(X.shape[0])
    X = X[p]
    Y = Y[p]

    print(X.shape, Y.shape)

    global X_train,Y_train,X_test,Y_test

    Y_new = DLModel.to_one_hot(22,Y)
    m = int(X.shape[0]*0.1) #number of test

    X_train , Y_train = X[m:].T , DLModel.to_one_hot(22,Y[m:])
    X_test , Y_test = X[:m].T, DLModel.to_one_hot(22,Y[:m])
    print(X_train)
    print(Y_train)

def generateNewImages(action, degrees = 15):
    dataSizePer = 96
    letters = {1:'א', 2:"ב",3:"ג",4:"ד",5:"ה",6:"ו",7:"ז",8:"ח",9:"ט",10:"י",11:"כ",12:"ל",13:"מ",14:"נ",15:"ס",16:"ע",17:"פ",18:"צ",19:"ק",20:"ר",21:"ש",22:"ת"}
    inv_map = {v: k for k, v in letters.items()}
    whiteBgImg = Image.open(r"SAVEDIMAGES\LETTERS\bg\0.jpg")
    for i in range(1,len(letters)+1): 
        saveIdx = dataSizePer
        for j in range(dataSizePer):
            img_path = r"SAVEDIMAGES\LETTERS\{lete}\{idx}.jpg".format(lete=letters[i], idx=j)
            image = Image.open(img_path)
            if action == 'rotate':
                newimg1 = image.rotate(degrees, fillcolor='white')
                newimg2 = image.rotate(360 - degrees, fillcolor='white')

                newimg1.save(r"SAVEDIMAGES\LETTERS\{lete}\{idx}.jpg".format(lete=letters[i], idx=saveIdx))
                saveIdx+=1

                newimg2.save(r"SAVEDIMAGES\LETTERS\{lete}\{idx}.jpg".format(lete=letters[i], idx=saveIdx))
                saveIdx+=1
            
def trainNetwork():
    model = DLModel()
    model.add(DLLayer("hidden", 256, (28*28,), "leaky_relu", "Xaviar", 0.1,regularization="L2",optimization="momentum"))
    model.add(DLLayer("output", 128, (256,), "leaky_relu", "Xaviar", 0.1,regularization="L2",optimization="momentum"))
    model.add(DLLayer("output", 64, (128 ,), "leaky_relu", "Xaviar", 0.1,regularization="L2",optimization="momentum"))
    model.add(DLLayer("output", 64, (64 ,), "leaky_relu", "Xaviar", 0.1,regularization="L2",optimization="momentum"))
    model.add(DLLayer("output", 22, (64 ,), "trim_softmax", "Xaviar", 0.1))

    model.compile("categorical_cross_entropy")

    costs = model.train(X_train, Y_train, 200)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(0.1))
    plt.show() 


    print("Train:")
    model.confusion_matrix(X_train, Y_train)
    print("Test:")
    test_accu = model.confusion_matrix(X_test, Y_test)

    model.save_weights(r"E:\Program Files\vscode\school project\SAVEDNETWORKS\{percent}".format(percent=round(test_accu*100, 2)))

def loadNetworkFromFile(fileName):
    model = DLModel()
    model.add(DLLayer("hidden", 256, (28*28,), "leaky_relu", "Xaviar", 0.1,regularization="L2",optimization="momentum"))
    model.add(DLLayer("output", 128, (256,), "leaky_relu", "Xaviar", 0.1,regularization="L2",optimization="momentum"))
    model.add(DLLayer("output", 64, (128 ,), "leaky_relu", "Xaviar", 0.1,regularization="L2",optimization="momentum"))
    model.add(DLLayer("output", 64, (64 ,), "leaky_relu", "Xaviar", 0.1,regularization="L2",optimization="momentum"))
    model.add(DLLayer("output", 22, (64 ,), "trim_softmax", "Xaviar", 0.1))

    model.compile("categorical_cross_entropy")

    for i in range(1, len(model.layers)):
        model.layers[i].init_weights("SAVEDNETWORKS\{name}\Layer{num}.h5".format(name=fileName, num=str(i)))

    return model

def predictLetterByPath(imagePath):
    image = Image.open(imagePath)
    image28 = image.resize((num_px, num_px), Image.Resampling.LANCZOS)
    gray_image = ImageOps.grayscale(image28)
    my_image = np.reshape(gray_image,(num_px*num_px,1 ))
    plt.imshow(my_image.reshape(28,28), cmap = matplotlib.cm.binary)
    plt.axis("off")
    my_image = my_image/255 - 0.5
    lastTur.Flag = True
    print(f"model predicts {prediction_dict[np.where(model.predict(my_image) == 1)[0][0]+1]}")
    return letters[np.where(model.predict(my_image) == 1)[0][0]+1] ## add letter to the string

def drawTextOnCanvas(removeLast = False):
    temp = Turtle('turtle')
    temp.hideturtle()
    temp.speed('fastest')
    temp.pensize(3)
    temp.pencolor("black")
    if len(string.st) > 1:
        if string.st[-2] == ' ':
            print("CLEAR")
            lastTur.lastTurtle.clear()
    if removeLast and lastTur.Flag:
        lastTur.lastTurtle.clear()
        string.st = string.st[:-1]
        lastTur.Flag = False
    temp.penup()
    temp.setposition(235,-160)
    temp.pendown()
    
    temp.write(string.st ,align="right",font=("Times New Roman",30,"normal"))
    temp.hideturtle()
    lastTur.lastTurtle = temp
    
def updateSTR(char=None,path = None):
    if char == None and path != None:
        string.st += predictLetterByPath(path)
        drawTextOnCanvas()
    elif char != None and path == None:
        if char == ' ':
            print("SPCAE")
            if string.st[-1] in letterToFinal.keys():
                newSt = string.st[:-1]
                newSt += letterToFinal[string.st[-1]]
                string.st = newSt
            string.st += '  '           
        else:
            string.st += char
            drawTextOnCanvas()
    else:
        raise Exception('Invalid Update')
    #print(f"updating str to + {prediction_dict[inv_map[string.st[-1]]]}")
    
 
np.random.seed(1)
num_px = 28
letters = {1:'א', 2:"ב",3:"ג",4:"ד",5:"ה",6:"ו",7:"ז",8:"ח",9:"ט",10:"י",11:"כ",12:"ל",13:"מ",14:"נ",15:"ס",16:"ע",17:"פ",18:"צ",19:"ק",20:"ר",21:"ש",22:"ת"}
prediction_dict = {1:'alef', 2:"bet",3:"gimel", 4:"daled",5:"hei",6:"vav",7:"zain",8:"heit",9:"tet",10:"yod",11:"caph",12:"lamed",13:"mem",14:"non",15:"sameh",16:"aain",17:"pei",18:"chadik",19:"kofh",20:"reish",21:"shin",22:"tafh"}
inv_map = {v: k for k, v in letters.items()}
letterToFinal = {'מ':"ם", 'נ':"ן","פ":"ף","צ":"ץ","כ":"ך"}
model = loadNetworkFromFile(99.05)

LETTER ="bg"
turtle.title("Letter detection")
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



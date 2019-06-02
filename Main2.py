import datetime #time display
import pandas as pd #for data anlaysis
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle #shuffling the data into colums row

import numpy as np
from bokeh.plotting import * #for grph ploting
from bokeh.layouts import column

from tkinter import *
import tkinter.messagebox

class UI:
    def __init__(self,w):
        self.window = w
        self.dataset = None
        self.window.title("Simplified Machine Learning")
        self.window.geometry("700x600+400+100")
        label1 = Label(self.window, text="SML Using Linear Regression Algorithm", fg="Black",bg="pink",font=("Calibri", 26, "bold")).pack(padx=10, pady=10)

        self.Label_Gen("Load the Dataset in the Program","Calibri",12,"italic",5,90)


        self.Button_Gen(20,"Load Data","Calibri",12,"bold",240,83,self.ReadData)
        self.txt = StringVar()
        self.user_x=None



        now = datetime.datetime.now()

        self.botFrame = Frame(self.window)
        self.botFrame.pack(side=BOTTOM)
        status2 = Label(self.botFrame, text=now.strftime('%H:%M:%S %A, %b %dth, %Y'), bd=1,bg="pink", relief=SUNKEN, anchor=E,font=("Calibri", 12, "italic")).pack(fill=X,side=RIGHT)



    def Label_Gen(self,txt,fnt,fontsize,style,xcord,ycord):
        label = Label(self.window, text=txt,bg="pink", font=(fnt,fontsize,style)).place(x=xcord,y=ycord)

    def Button_Gen(self,width,txt,fnt,fontsize,style,xcord,ycord, Events):
        btn = Button(self.window, text=txt, relief=GROOVE, bg="pink", font=(fnt,fontsize,style),command=Events).place(x=xcord,y=ycord)

    def showGraph(self):
        LR.plotGraph()

    def ReadData(self):
        try:
            #reading dataset csv file:
            self.dataset = pd.read_csv("CarData.csv")
            #shuffling the data for better division
            self.dataset = shuffle(self.dataset)
            tkinter.messagebox.showinfo("Success", "Data Loaded Successfully")
            self.status = Label(self.botFrame, text="Status: Training and Testing of Machine Completed!!!      ", bd=1, relief=SUNKEN, anchor=W,font=("Calibri", 12, "italic")).pack(fill=X,side=LEFT)
            self.Label_Gen("DataSet Information: ", "Calibri", 12, "italic", 5, 150)
            self.Label_Gen(str(len(self.dataset))+" Records and 16 Columns", "Calibri", 12, "italic", 150, 150)

            self.Label_Gen("Training Set Consists of 75% (6,054 records) &\nTesting Set Consists of 25% (2,018 records)","Calibri", 12, "italic", 5, 180)

            #division of data in train n test
            LR.divide_into_train_test_set()
            self.Button_Gen(20, "Fit the Linear Regression model to DataSet", "Calibri", 12, "bold", 20, 240, LR.fiting_the_Reg_model)


        except:
            tkinter.messagebox.showinfo("Error","Unable to Read Data!")
            print("Unable to Read Data!")

    #input from text box on button click

    def ReadEst(self):
        try:
            st = self.txt.get()
            self.user_x=int(st)
            y=LR.predict_user_x(self.user_x)
            tkinter.messagebox.showinfo("Info","Predicted Price : "+str(y))
        except:
            tkinter.messagebox.showinfo("Error","Unable to Read Data!")
            print("Unable to Read Data!")

    def getDataSet(self):
        return self.dataset

class LinearReg:
    def __init__(self):
        self.Dataset=None
        self.y_predict1=None
        self.MSE = None         #Mean Squared Error
        self.Accuracy = None

    def divide_into_train_test_set(self):
        self.Dataset=ui.getDataSet()

        # Y Values:   Manufacturer's Suggested Retail Price (MSRP) column (L) 11
        # X Values:   Engine HP column (G) 6

        self.x_values = self.Dataset.values[:, 6]
        self.y_values = self.Dataset.values[:, 11]

        # Dividing Into_train_test arrays
        max_index = int(len(self.Dataset) * 0.75)


        self.training_set_x = self.x_values[0:max_index]
        self.training_set_y = self.y_values[0:max_index]

        self.testing_set_x = self.x_values[max_index:]
        self.testing_set_y = self.y_values[max_index:]

        self.reg_model = LinearRegression()


    #main calculation on button:"Fit the Linear Regression model to DataSet" click event this method called
    def fiting_the_Reg_model(self):

        #fitting the regression line

        self.reg_model.fit(self.training_set_x.reshape(-1, 1), self.training_set_y)
        y_predict = self.reg_model.predict(self.testing_set_x.reshape(-1, 1))

        self.y_predict1=np.array([y_predict])
        self.y_predict1=self.y_predict1.ravel() #converting 2d array y_predict1 into 1 d array using ravel() for grahph ploting

        SSE = 0
        for i in range(0, len(self.y_predict1)):
            diff = self.testing_set_y[i] - self.y_predict1[i]
            diff = diff * diff
            SSE = SSE + diff

        # calculating mean squared error and Accuracy

        self.Accuracy = self.reg_model.score(self.testing_set_x.reshape(-1, 1),self.testing_set_y)*100

        self.MSE = SSE / len(self.training_set_x)



        ui.Label_Gen("Accuracy :\t\t "+str(self.Accuracy), "Calibri", 12, "italic", 5, 290)

        ui.Label_Gen("Mean Squared Error: " + str(self.MSE), "Calibri", 12, "italic", 5, 330)

        ui.Button_Gen(20, "Show Graph", "Calibri", 12, "bold", 240, 380, ui.showGraph)

        ui.status = None
        ui.status=Label(ui.botFrame, text="Status: Fitting of Linear Regression Model Completed!!!       ", bd=1,
                       relief=SUNKEN, anchor=W, font=("Calibri", 12, "italic")).pack(fill=X, side=LEFT)

        ui.Label_Gen("Enter Value for Engine HP est:", "Calibri", 12, "italic", 5, 426)
        ui.txt01 = Entry(ui.window, width=30, textvariable=ui.txt).place(x=220, y=430)
        ui.Button_Gen(20, "Predict", "Calibri", 12, "bold", 240, 470, ui.ReadEst)

    def predict_user_x(self,user_x):
        x=user_x
        y_cap = self.reg_model.intercept_ + self.reg_model.coef_*user_x
        return y_cap



    def plotGraph(self):
        output_file("Linear_Regression.html")
        plot = figure(plot_width=1100, plot_height=600, x_axis_label="Engine HP (Independent)",
                      y_axis_label="Price (Dependent)", title="Training Set")
        plot2 = figure(plot_width=1100, plot_height=600, x_axis_label="Engine HP (Independent)",
                       y_axis_label="Price (Dependent)", title="Testing Set")
        plot3 = figure(plot_width=1100, plot_height=600, x_axis_label="Engine HP (Independent)",
                       y_axis_label="Price (Dependent)", title="Linear Regression")

        plot.scatter(self.training_set_x, self.training_set_y, marker="square", color="blue")

        plot2.scatter(self.testing_set_x, self.testing_set_y, marker="circle", color="green")
        plot3.circle(self.testing_set_x, self.y_predict1)
        plot3.line(self.testing_set_x, self.y_predict1, color="black")

        show(column(plot, plot2, plot3))


#Y Values:   Manufacturer's Suggested Retail Price (MSRP) column (L) 11
#X Values:   Engine HP column (G) 6
#
window=Tk()
ui = UI(window)
LR = LinearReg()
window.mainloop()



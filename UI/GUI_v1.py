#FaceOff Deepfake Detector GUI
#Lead Developer: Gabriel Schmitt
#Contributions by: Margo Sikes, Caleb Graham

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.ttk import Progressbar
from os import path
import itertools
import os

import threading
import queue

import cnnClassifier
import svmClassifier

files = []
predictions = []
res = []

#This class handles basic layout and switching between frames
class GUI(Tk):

    def __init__(self,*args,**kwargs):

        Tk.__init__(self, *args,**kwargs)
        container = Frame(self)

        #Program icon to appear in the upper left of the window
        Tk.iconbitmap(self, default="SP_Icon.ico")

        self.title("FaceOff Deepfake Detector")

        #Packing container the frames go into
        container.pack(side="top", fill="both", expand=True)

        self.geometry('595x330')
        #self.minsize(595,330)
        #self.maxsize(595,330)
        
        #Section contributed by Margo Sikes
        global files
        global cnnQue
        global cnnThread
        global svmQue
        global svmThread

        global selectState

        global var
        global var2
        var = StringVar(value=files)
        var2 = StringVar(value=res)

        cnnQue = queue.Queue()
        cnnThread = threading.Thread(target=lambda q, arg1: q.put(cnnClassifier.classifier(arg1)), args=(cnnQue,files))
        svmQue = queue.Queue()
        svmThread = threading.Thread(target=lambda q, arg1: q.put(svmClassifier.classifier(arg1)), args=(svmQue,files))

        #Dictionary where all pages will go
        self.frames = {}
        
        for F in (StartPage, AlgPage, ProgBarPage, ResultsPage):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    #Raises the next frame on top of the current one
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

#The page users will see when they first start the program
class StartPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        
        lbl = Label(self, text="Deepfake Detector", font=("Arial Bold", 30))
        lbl.grid(column=2, row=0, ipadx=20)

        verNbr = Label(self, text="Version 1.0.0")
        verNbr.grid(column=3,row=3)

        def clickedFile():

            # Askopenfilenames returns something that is not in the correct format, so must append to list
            filesUnformatted = filedialog.askopenfilenames(title="Import Files", filetype=(("MP4 Files","*.mp4"),))
            for item in filesUnformatted:
                files.append(item)
            print(files)
                                      
        def clickedLog():

            clr = messagebox.askyesno("Clear Log", "Are you sure you want to clear the log? This action is irreversible")

            if clr == True:
                print("Log Cleared")
            if clr == False:
                print("Log Not Cleared")

        def clickedNext():
            
            nxt = messagebox.askyesno("Next", "Have you imported all your desired files?")
            if nxt == True:
                controller.show_frame(AlgPage)
            if nxt == False:
                controller.show_frame(StartPage)
                
        #Buttons and their commands
        btn = ttk.Button(self, text="Import Files",width=12, command=clickedFile)

        btn1 = ttk.Button(self, text="Clear Log", width=12,command=clickedLog)

        btn2 = ttk.Button(self, text="Next", width=12, command=clickedNext)

        #Placing buttons
        btn.grid(column=0, row=1, padx=10, pady=15)

        btn1.grid(column=0, row=2, padx=10, pady=15)

        btn2.grid(column=0, row=3, padx=10, pady=15)

        logo = PhotoImage(file='SP_Mascot.png')
        logo.image = logo
        labelLogo = Label(self, image=logo)

        labelLogo.grid(row=2, column=2)

class AlgPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        
        lbl = Label(self, text="Algorithms", font=("Arial Bold", 30))
        lbl.grid(column=2, row=0, ipadx=50)

        verNbr = Label(self, text="Version 1.0.0")
        verNbr.grid(column=3,row=3)

        def c1hk():
            chk1_state.set(True)

            chk2_state.set(False)

        def c2hk():
            chk1_state.set(False)

            chk2_state.set(True)
        
        def nextPage():

            global selectState
            if chk1_state.get() == 1:
                npage = messagebox.askyesno("Run Program","Run program with the Convolutional Neural Network?")
                if npage == True:   
                    controller.show_frame(ProgBarPage)
                    selectState = 0
                    cnnThread.start()
            elif chk2_state.get() == 1:
                npage = messagebox.askyesno("Run Program","Run program with the Support Vector Machine?")
                if npage == True:
                    controller.show_frame(ProgBarPage)
                    selectState = 1
                    svmThread.start()
            else: 
                messagebox.showwarning("Must select an algorithm","Please select an algorithm to continue")

        #Setting up radio buttons

        chk1_state = BooleanVar() 
        chk2_state = BooleanVar()
        
        #Giving the radio buttons variables to reference and properties
        
        chk1 = ttk.Radiobutton(self, text="Conv. Neural Network  ",value=1, command = c1hk) 

        chk2 = ttk.Radiobutton(self, text="Support Vector Machine",value=2, command = c2hk)

        #Placing radio buttons
        chk1.grid(column=0, row=1, padx=0, pady=30, columnspan=2)

        chk2.grid(column=0, row=2, padx=0, pady=15, columnspan=2)
        

        nxt = ttk.Button(self, text="Next", width=10, command=nextPage) #next button

        nxt.grid(column=2, row=3)

        #This will provide appropriate spacing without the image appearing
        logo = PhotoImage(file='SP_Mascot.png')
        labelLogo = Label(self, image=logo)

        labelLogo.grid(row=2, column=2)

#The page users will see while the program processes the video
class ProgBarPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        lbl = Label(self, text="Deepfake Detector", font=("Arial Bold", 30))
        lbl.grid(column=2, row=1)

        verNbr = Label(self, text="Version 1.0.0")
        verNbr.grid(column=2,row=5, padx=(500,1))
        
        loadingLbl = Label(self, text="Loading...", font=("Arial Bold", 20))
        loadingLbl.grid(column=1,row=3,pady=10,columnspan=2)

        #Section contributed by Margo Sikes
        def nextPage():
            global predictions
            global selectState
            if selectState == 0:
                cnnThread.join()
                predictions = cnnQue.get()
            if selectState == 1:
                svmThread.join()
                predictions = svmQue.get()
            results = convertResults(predictions)
            combinedResults = []
            video_name = ""
            for i in range(len(files)):
                video_name = os.path.splitext(os.path.basename(files[i]))[0]
                combinedResults.append(video_name+': '+results[i])
            res = ""
            for item in combinedResults:
                res = res + item
            var.set(files)
            var2.set(res)
            controller.show_frame(ResultsPage)
        
        #Section contributed by Margo Sikes
        def convertResults(predictions):
            results = []
            for prediction in predictions:
                if prediction < 0.2:
                    results.append("Likely real ({0:.2f})\n".format(prediction))
                elif prediction >= 0.2 and prediction < 0.5:
                    results.append("Probably real ({0:.2f})\n".format(prediction))
                elif prediction == 0.5:
                    results.append("Not sure ({0:.2f})\n".format(prediction))
                elif prediction > 0.5 and prediction < 0.8:
                    results.append("Probably fake ({0:.2f})\n".format(prediction))
                else:
                    results.append("Likely fake: ({0:.2f})\n".format(prediction))
            return results

        nxt = ttk.Button(self, text="Next", width=10, command=nextPage)

        nxt.grid(column=2,row=4, padx=250)
        logo = PhotoImage(file='SP_Mascot.png')
        labelLogo = Label(self, image=logo)

        labelLogo.grid(row=2, column=2)
        
class ResultsPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        lbl = Label(self, text="Results", font=("Arial Bold", 30))
        lbl.grid(column=2, row=0, ipadx=20)

        verNbr = Label(self, text="Version 1.0.0")
        verNbr.grid(column=3,row=3)

        def restartBtn():
            res = messagebox.askyesno("Restart", "Restart from the algorithm page?")
            if res == True:
                    controller.show_frame(AlgPage)

        #Contributed by Caleb Graham
        def extBtn():
            ext = messagebox.askyesno("Exit", "Are you sure you would like to exit the program?")
            if ext == True:
                exit()

        restart = ttk.Button(self, text="Restart", width=10, command=restartBtn)

        export = ttk.Button(self, text="Export Results", width=15)

        ext = ttk.Button(self, text="Exit", width=10, command=extBtn)

        restart.grid(row=2, column=1, padx=48, pady=35)

        export.grid(row=2, column=2, padx=75)

        ext.grid(row=2, column=3, padx=60)
        
        result = Label(self, textvariable=(var2))

        result.grid(row=1, column=2, pady=15)
        
app = GUI()
app.mainloop()

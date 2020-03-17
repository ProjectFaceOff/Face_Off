from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from os import path

#This class handles basic layout and switching between frames. It breaks if you look at it wrong.
class Detector(Tk):

    def __init__(self,*args,**kwargs):

        Tk.__init__(self, *args,**kwargs)
        container = Frame(self)

        self.title("FaceOff Deepfake Detector")

        container.pack(side="top", fill="both", expand=True)
        #container.grid(row=0, column=0)

        self.geometry('595x330')
        self.minsize(595,330)
        self.maxsize(595,330)

        #Dictionary where all pages will go
        self.frames = {}

        for F in (StartPage, AlgPage, ProgBarPage, ResultsPage):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()


#The page users will see when they first start the program
class StartPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        
        lbl = Label(self, text="Deepfake Detector", font=("Arial Bold", 30))
        lbl.grid(column=2, row=0, ipadx=20)

        verNbr = Label(self, text="Version 0.0.0.0")
        verNbr.grid(column=3,row=3)

#Above code should be copied for every subsequent page

        def clickedFile():

            file = filedialog.askopenfilename(title="Import Files", filetype=(("MP4 Files","*.mp4"),))
            print(file) #print for testing purposes, will return later
                                      
        def clickedLog():

            clr = messagebox.askyesno("Clear Log", "Are you sure you want to clear the log? This action is irreversible")

            if clr == True:
                print("Log Cleared")
            if clr == False:
                print("Log Not Cleared")
            #print for testing purposes. will attach to functions that clear a specific folder later

        def clickedNext():

            nxt = messagebox.askyesno("Next", "Have you imported all your desired files?")

            if nxt == True:
                controller.show_frame(AlgPage)
            if nxt == False:
                controller.show_frame(StartPage)


        #Buttons and their functions
        btn = Button(self, text="Import Files",width =10, command=clickedFile)

        btn1 = Button(self, text="Clear Log", width=10,command=clickedLog)

        btn2 = Button(self, text="Next", width=10, command = clickedNext)

        #Button layout
        btn.grid(column=0, row=1, padx=10, pady=15)

        btn1.grid(column=0, row=2, padx=10, pady=15)

        btn2.grid(column=0, row=3, padx=10, pady=15)

        #Placing our logo - not sure why it doesn't work
        logo = PhotoImage(file='SP_Mascot.png')
        labelLogo = Label(self, image=logo)

        labelLogo.grid(row=2, column=2)

class AlgPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        
        lbl = Label(self, text="Algorithms", font=("Arial Bold", 30))
        lbl.grid(column=2, row=0, ipadx=60)

        verNbr = Label(self, text="Version 0.0.0.0")
        verNbr.grid(column=3,row=3)

        chk1_state = BooleanVar() #setting up checkbuttons

        chk2_state = BooleanVar()

        chk3_state = BooleanVar()

        chk1_state.set(False) #defaulting the box to unchecked

        chk2_state.set(False)

        chk3_state.set(False)

        chk1 = Checkbutton(self, text="Al Gore Rhythm 1", var=chk1_state) #giving the check boxes variables to reference and properties

        chk2 = Checkbutton(self, text="Al Gore Rhythm 2", var=chk2_state)

        chk3 = Checkbutton(self, text="Al Gore Rhythm 3", var=chk3_state)

        chk1.grid(column=0, row=1, padx=10, pady=15)

        chk2.grid(column=0, row=2, padx=10, pady=15)

        chk3.grid(column=0, row=3, padx=10, pady=15)

        logo = PhotoImage(file='SP_Mascot.png')
        labelLogo = Label(self, image=logo)

        labelLogo.grid(row=2, column=2)

class ProgBarPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        lbl = Label(self, text="Deepfake Detector", font=("Arial Bold", 30))
        lbl.grid(column=2, row=0, ipadx=20)

        verNbr = Label(self, text="Version 0.0.0.0")
        verNbr.grid(column=3,row=3)

class ResultsPage(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        lbl = Label(self, text="Deepfake Detector", font=("Arial Bold", 30))
        lbl.grid(column=2, row=0, ipadx=20)

        verNbr = Label(self, text="Version 0.0.0.0")
        verNbr.grid(column=3,row=3)

app = Detector()
app.mainloop()

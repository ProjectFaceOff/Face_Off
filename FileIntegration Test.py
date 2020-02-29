from tkinter import *

from tkinter import filedialog

#from os import path

from tkinter import messagebox

window = Tk()

window.title("FaceOff Deepfake Detector")

lbl = Label(window, text="Yes", font=("Arial Bold", 30))

lbl.grid(column=0, row=0)

window.geometry('300x300')

def clickedFile():

    file = filedialog.askopenfilename()

def clickedLog():

    messagebox.askyesno("Clear Log", "Are you sure you want to clear the log? This action is irreversible")

def clickedNext():

    messagebox.showerror("No files selected", "Please select files")

btn = Button(window, text="Import Files", command=clickedFile)

btn1 = Button(window, text="Clear Log", command=clickedLog)

btn2 = Button(window, text="Next", command=clickedNext)

btn.grid(column=0, row=1)

btn1.grid(column=0, row=2)

btn2.grid(column=0, row=3)

window.mainloop()



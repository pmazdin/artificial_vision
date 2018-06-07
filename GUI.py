import Tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
import os
import shutil

global last_frame
global cap
global label
global frame
global pic

def change_label(text, done=False):
    global frame
    global label
    global pic
    label.destroy()
    if not done:
        write_text = "      " + text
    else:
        write_text = "      " + text + " DONE"
    label = tk.Label(frame, text = write_text,foreground="red",font=("Helvetica", 16))
    label.pack(side = tk.RIGHT)

def take_photo(name, count):
    global pic
    cv2.imwrite(name + "/" + str(count) + ".jpg",pic)
    


def train():
    global frame
    global label
    global pic
    print("Train chosen!")
    if label is not None:
        label.destroy()
    label = tk.Label(frame, text ="     LOOK LEFT",foreground="red",font=("Helvetica", 16))
    label.pack(side = tk.RIGHT)

    directory = ["training_left", "training_right", "training_straight"]
    command = ["LOOK RIGHT", "LOOK STRAIGHT"]

    for i in range(len(directory)):

        if os.path.exists(directory[i]):
            shutil.rmtree(directory[i])
        os.makedirs(directory[i])

        for j in range(15):
            root.after(j*1000, take_photo, directory[i], j)

        if (i+1) < len(directory):
            root.after(i*5000, change_label, command[i])


    root.after(15000, change_label, "TRAINING", True)



def authorize():
    global frame
    global label
    global pic
    print("Authorization chosen!")
    if label is not None:
        label.destroy()
    label = tk.Label(frame, text ="     LOOK STRAIGHT",foreground="red",font=("Helvetica", 16))
    label.pack(side = tk.RIGHT)
    root.after(5000, change_label, "BLINK FEW TIMES")
    root.after(10000, change_label, "AUTHORIZATION", True)

def load():
    print("Load chosen!")

def save():
    print("Save chosen!")



def show_video():

    if not cap.isOpened():
        print("ERROR: cannot open the camera")

    flag, frame = cap.read()

    if flag is None:
        print("ERROR: cannot read the camera!")
    elif flag:
        global last_frame
        global pic
        last_frame = frame.copy()
        pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB) 
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_video)

     



root = tk.Tk()
root.title('Choose an action')
root.geometry('700x500')
frame = tk.Frame(root)
frame.pack()

label = None

button = tk.Button(frame, text="Train", fg="red", command=train)
button.pack(side=tk.LEFT)

button = tk.Button(frame, text="Authorize", fg="blue", command=authorize)
button.pack(side=tk.LEFT)

button = tk.Button(frame, text="Load", fg="green", command=load)
button.pack(side=tk.LEFT)

button = tk.Button(frame, text="Save", fg="yellow", command=save)
button.pack(side=tk.LEFT)

                    
last_frame = np.zeros((280, 340, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0)
lmain = tk.Label(master=root)
lmain.pack(side = tk.BOTTOM)
show_video()


root.mainloop()


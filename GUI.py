import Tkinter as tk
import cv2
from PIL import Image, ImageTk
from sklearn.datasets import load_files
from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

global last_frame
global cap
global label
global frame
global pic



def pca():
    # Construct the input matrix
    in_matrix = None
    directory = "train_straight"
    for f in os.listdir(directory):
        im = cv2.imread(os.path.join(directory, f), cv2.IMREAD_GRAYSCALE)

        # vec = im.reshape(w * h)

        try:
            in_matrix = np.vstack((in_matrix, im))
        except:
            in_matrix = im

    if in_matrix is not None:
    #     X = in_matrix.data
    #     y = in_matrix.target
    #     mean, eigenvectors = cv2.PCACompute(in_matrix, np.mean(in_matrix, axis=0).reshape(1, -1))
    #
    # print(eigenvectors)
        pca = PCA(n_components=50, whiten=True).fit(in_matrix)
        X_train_pca = pca.transform(in_matrix)
        plt.figure()
        plt.plot(pca.singular_values_)
        plt.show()

    c = svm.SVC()
    # c.fit(X_train, y_train)
    c.fit(X_train_pca, 0)




def check_model_exists(path):

    ret_value = False
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".model"):
                    ret_value = True

    return ret_value

def change_label(text, done=False, button_load_set=False, button_save_set=False):
    global frame
    global label
    global pic
    label.destroy()
    if not done:
        write_text = "   " + text
    else:
        write_text = "   " + text + " DONE"
        if text == "TRAINING":
            button_authorize['state'] = tk.NORMAL
            button_save['state'] = tk.NORMAL
            button_authorize['state'] = tk.NORMAL
        else:
            button_load['state'] = button_load_set
            button_save['state'] = button_save_set
            button_authorize['state'] = tk.NORMAL
    label = tk.Label(frame, text = write_text,foreground="red",font=("Helvetica", 10))
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
    label = tk.Label(frame, text ="  LOOK LEFT",foreground="red",font=("Helvetica", 10))
    label.pack(side = tk.RIGHT)

    directory = ["train_left", "train_right", "train_straight"]
    command = ["LOOK RIGHT", "LOOK STRAIGHT"]

    button_authorize['state'] = tk.DISABLED

    for i in range(len(directory)):

        if os.path.exists(directory[i]):
            shutil.rmtree(directory[i])
        os.makedirs(directory[i])

        for j in range(15):
            root.after(j*1000, take_photo, directory[i], j)

        if (i+1) < len(directory):
            root.after(i*5000, change_label, command[i])


    root.after(15000, change_label, "TRAINING", True)


def close():
    root.destroy()


def authorize():
    global frame
    global label
    global pic
    print("Authorization chosen!")
    if label is not None:
        label.destroy()
    button_load_state = button_load['state']
    button_save_state = button_save['state']
    button_load['state'] = tk.DISABLED
    button_save['state'] = tk.DISABLED
    button_authorize['state'] = tk.DISABLED
    label = tk.Label(frame, text ="  LOOK STRAIGHT",foreground="red",font=("Helvetica", 10))
    label.pack(side = tk.RIGHT)

    directory = ["authorize_straight", "", "authorize_right", "authorize_left"]
    command = ["BLINK FEW TIMES", "LOOK RIGHT", "LOOK LEFT"]

    for i in range(len(directory)):
        if (directory[i] != ""):
            if os.path.exists(directory[i]):
                shutil.rmtree(directory[i])
            os.makedirs(directory[i])

            for j in range(15):
                root.after(j * 1000, take_photo, directory[i], j)

        if (i + 1) < len(directory):
            root.after((i+1) * 5000, change_label, command[i])


    # root.after(5000, change_label, "BLINK FEW TIMES")

    root.after(20000, change_label, "AUTHORIZATION", True, button_load_state, button_save_state)


def load():
    print("Load chosen!")
    button_load['state'] = tk.DISABLED
    button_authorize['state'] = tk.NORMAL


def save():
    print("Save chosen!")
    button_load['state'] = tk.NORMAL
    button_save['state'] = tk.DISABLED
    directory = "train_model"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    np.savetxt(directory + '/straight_pca.model', straight_pca)
    np.savetxt(directory + '/straight_sift.model', straight_sift)
    np.savetxt(directory + '/left_pca.model', left_pca)
    np.savetxt(directory + '/left_sift.model', left_sift)
    np.savetxt(directory + '/right_pca.model', right_pca)
    np.savetxt(directory + '/right_sift.model', right_sift)






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
        pic = cv2.resize(pic, (500, 370))
        img = Image.fromarray(pic)

        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_video)



root = tk.Tk()
root.title('Choose an action')
root.geometry('600x380')
frame = tk.Frame(root)
frame.pack()

label = None

straight_pca = np.zeros(5)
straight_sift = np.zeros(5)
left_pca = np.zeros(5)
left_sift = np.zeros(5)
right_pca = np.zeros(5)
right_sift = np.zeros(5)

button_train = tk.Button(frame, text="Train", fg="red", command=train)
button_train.pack(side=tk.LEFT)

button_authorize = tk.Button(frame, text="Authorize", fg="blue", command=authorize, state=tk.DISABLED)
button_authorize.pack(side=tk.LEFT, padx=5)

if check_model_exists("train_model"):
    button_load = tk.Button(frame, text="Load", fg="green", command=load)
else:
    button_load = tk.Button(frame, text="Load", fg="green", command=load, state=tk.DISABLED)

button_load.pack(side=tk.LEFT)

button_save = tk.Button(frame, text="Save", fg="yellow", command=save, state=tk.DISABLED)
button_save.pack(side=tk.LEFT, padx=5)

button_close = tk.Button(frame, text="Close", fg="black", command=close)
button_close.pack(side=tk.RIGHT, padx=30)


last_frame = np.zeros((280, 340, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0)
lmain = tk.Label(master=root)
lmain.pack(side = tk.BOTTOM)
show_video()


root.mainloop()



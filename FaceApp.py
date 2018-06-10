import Tkinter as tk
import os
import cv2
import shutil
import numpy as np
from PIL import Image, ImageTk
from deep_head_pose.code import headposedlib
from facemodel import *
LARGE_FONT = ("Verdana", 12)
NORMAL_FONT = ("Verdana", 10)
SMALL_FONT = ("Verdana", 8)



# http://zetcode.com/gui/tkinter/dialogs/

# controller
class FaceApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        self.root = tk.Tk.__init__(self, *args, **kwargs)
        self.img_w = 480
        self.img_h = 360
        self.geometry(str(self.img_w + 60) + "x" + str(self.img_h*2 + 120) + "+300+300")
        tk.Tk.wm_title(self, "FaceApp")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}

        # add new pages to that list:
        #for F in (StartPage):
        frame = StartPage(container, self)
        self.frames[StartPage] = frame
        frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)
        self.txt = tk.Text(self)
        self.txt.pack(fill=tk.BOTH, expand=1)


    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def donothing(self):
        filewin = tk.Toplevel(self)
        button = tk.Button(filewin, text="Do nothing button")
        button.pack()
        print("do nothing")

    def onOpen(self):
        # https://stackoverflow.com/questions/16429716/opening-file-tkinter
        ftypes = [('csv files', '*.csv')]
        dlg = tk.filedialog.Open(self, filetypes = ftypes)
        fl = dlg.show()

        if fl != '':
            text = self.readFile(fl)
            self.txt.insert(tk.END, text)

    def readFile(self, filename):
        f = open(filename, "r")
        text = f.read()
        return text



class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.label_txt = tk.Label(self, text="Choose an action", font=LARGE_FONT)
        self.label_txt.pack(pady=10, padx=10)

        #button1 = tk.Button(self, text="Visit Page 1", command=lambda: controller.show_frame(PageOne))
        #button1.pack()

        self.button_train = tk.Button(self, text="Train", fg="red", command=self.train)
        self.button_train.pack(side=tk.LEFT, padx=5)

        self.button_authorize = tk.Button(self, text="Authorize", fg="blue", command=self.authorize, state=tk.DISABLED)
        self.button_authorize.pack(side=tk.LEFT, padx=5)

        self.button_load_model = tk.Button(self, text="Load Model", fg="green", command=self.load_model)
        self.button_load_model.pack(side=tk.LEFT)

        self.button_save_model = tk.Button(self, text="Save Model", fg="yellow", command=self.save_model, state=tk.DISABLED)
        self.button_save_model.pack(side=tk.LEFT, padx=5)

        self.button_close = tk.Button(self, text="Close", fg="black", command=controller.quit)
        self.button_close.pack(side=tk.LEFT, padx=5)


        self.img_cam = np.zeros((controller.img_h, controller.img_w, 3), dtype=np.uint8)
        self.img_res = None
        self.img_cur = None
        self.cap = cv2.VideoCapture(0)

        self.label_cam_txt = tk.Label(self, text="live camera stream", font=SMALL_FONT)
        self.label_cam_txt.pack(side=tk.TOP, pady=0, padx=10)

        self.label_cam_stream = tk.Label(master=controller)
        self.label_cam_stream.pack(side=tk.TOP, pady=5)

        self.label_img_res_txt = tk.Label(self, text="processed image:", font=SMALL_FONT)
        self.label_img_res_txt.pack(side=tk.TOP,pady=0, padx=10)
        self.label_img_res = tk.Label(master=controller)
        self.label_img_res.pack(side=tk.TOP, pady=5)

        self.model = FaceModel()

        self.show_video()




    def save_image(self, name, count):
        cv2.imwrite(name + "/" + str(count) + ".jpg", self.cur_img)

    def train(self):
        print("Train chosen!")
        self.button_save_model['state'] = tk.NORMAL

        self.model.train_model()


    def authorize(self):
        print("Authorization chosen!")

    def load_model(self):
        print("Load model chosen!")
        #self.button_load_model['state'] = tk.DISABLED
        self.button_save_model['state'] = tk.NORMAL
        self.button_authorize['state'] = tk.NORMAL

    def save_model(self):
        print("Load model chosen!")
        self.button_load_model['state'] = tk.DISABLED
        self.button_authorize['state'] = tk.NORMAL

    def show_video(self):
        if not self.cap.isOpened():
            print("ERROR: cannot open the camera")

        flag, img_new = self.cap.read()
        print("capture...")
        if flag is None:
            print("ERROR: cannot read the camera!")
        elif flag:
            self.img_cam = img_new.copy()
            self.img_cam = cv2.cvtColor(self.img_cam, cv2.COLOR_BGR2RGB)
            self.img_cam = cv2.resize(self.img_cam, (self.controller.img_w, self.controller.img_h))

            img1 = Image.fromarray(self.img_cam)
            imgtk_cam = ImageTk.PhotoImage(image=img1)
            self.label_cam_stream.imgtk = imgtk_cam
            self.label_cam_stream.configure(image=imgtk_cam)

            self.model.set_cam_image(self.img_cam)
            self.img_res = self.model.get_res_image()

            if self.img_res is None:
                self.img_res = self.img_cam

            img2 = Image.fromarray(self.img_res)
            imgtk_res = ImageTk.PhotoImage(image=img2)
            self.label_img_res.imgtk = imgtk_res
            self.label_img_res.configure(image=imgtk_res)

            self.label_cam_stream.after(50, self.show_video)


def main():
    app = FaceApp()
    app.mainloop()


if __name__ == '__main__':
    main()

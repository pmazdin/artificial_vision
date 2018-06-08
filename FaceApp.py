import Tkinter as tk
import os
import cv2
import shutil
import numpy as np
from PIL import Image, ImageTk
from deep_head_pose.code import headposedlib

LARGE_FONT = ("Verdana", 12)
NORMAL_FONT = ("Verdana", 10)
SMALL_FONT = ("Verdana", 8)



# http://zetcode.com/gui/tkinter/dialogs/

# controller
class FaceApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        self.root = tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "FaceApp")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}

        self.DHP = headposedlib.HeadPoseDLib("/home/jungr/workspace/NAV/development/face_authorization_py/deep_head_pose/hopenet_alpha2.pkl",
                                             "/home/jungr/workspace/NAV/development/face_authorization_py/deep_head_pose/mmod_human_face_detector.dat")


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

        self.label = tk.Label(self, text="Choose an action", font=LARGE_FONT)
        self.label.pack(pady=10, padx=10)

        #button1 = tk.Button(self, text="Visit Page 1", command=lambda: controller.show_frame(PageOne))
        #button1.pack()

        self.button_train = tk.Button(self, text="Train", fg="red", command=self.train)
        self.button_train.pack(side=tk.LEFT)

        self.button_authorize = tk.Button(self, text="Authorize", fg="blue", command=self.authorize, state=tk.DISABLED)
        self.button_authorize.pack(side=tk.LEFT, padx=5)

        if self.check_model_exists("train_model"):
            self.button_load = tk.Button(self, text="Load", fg="green", command=self.load)
        else:
            self.button_load = tk.Button(self, text="Load", fg="green", command=self.load, state=tk.DISABLED)

        self.button_load.pack(side=tk.LEFT)

        self.button_save = tk.Button(self, text="Save", fg="yellow", command=self.save, state=tk.DISABLED)
        self.button_save.pack(side=tk.LEFT, padx=5)

        self.button_close = tk.Button(self, text="Close", fg="black", command=controller.quit)
        self.button_close.pack(side=tk.RIGHT, padx=30)


        self.last_img = np.zeros((280, 340, 3), dtype=np.uint8)
        self.cur_img = None
        self.cap = cv2.VideoCapture(0)

        self.lmain = tk.Label(master=controller)
        self.lmain.pack(side=tk.BOTTOM)
        self.show_video()


    def check_model_exists(self, path):

        ret_value = False
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".model"):
                        ret_value = True

        return ret_value

    def change_label(self, text, done=False, button_load_set=False, button_save_set=False):
        self.label.destroy()
        if not done:
            write_text = "   " + text
        else:
            write_text = "   " + text + " DONE"
            if text == "TRAINING":
                self.button_authorize['state'] = tk.NORMAL
                self.button_save['state'] = tk.NORMAL
                self.button_authorize['state'] = tk.NORMAL
            else:
                self.button_load['state'] = button_load_set
                self.button_save['state'] = button_save_set
                self.button_authorize['state'] = tk.NORMAL

        self.label = tk.Label(self, text=write_text, foreground="red", font=NORMAL_FONT)
        self.label.pack(side=tk.RIGHT)

    def take_photo(self, name, count):
        cv2.imwrite(name + "/" + str(count) + ".jpg", self.cur_img)

    def train(self):
        print("Train chosen!")
        if self.label is not None:
            self.label.destroy()

        self.label = tk.Label(self, text="  LOOK LEFT", foreground="red", font=NORMAL_FONT)
        self.label.pack(side=tk.RIGHT)

        directory = ["train_left", "train_right", "train_straight"]
        command = ["LOOK RIGHT", "LOOK STRAIGHT"]

        self.button_authorize['state'] = tk.DISABLED

        for i in range(len(directory)):

            if os.path.exists(directory[i]):
                shutil.rmtree(directory[i])
            os.makedirs(directory[i])

            for j in range(15):
                self.controller.after(j * 1000, self.take_photo, directory[i], j)

            if (i + 1) < len(directory):
                self.controller.after(i * 5000, self.change_label, command[i])

        self.controller.after(15000, self.change_label, "TRAINING", True)

    def authorize(self):
        print("Authorization chosen!")
        if self.label is not None:
            self.label.destroy()
        button_load_state = self.button_load['state']
        button_save_state = self.button_save['state']
        self.button_load['state'] = tk.DISABLED
        self.button_save['state'] = tk.DISABLED
        self.button_authorize['state'] = tk.DISABLED
        self.label = tk.Label(self, text="  LOOK STRAIGHT", foreground="red", font=NORMAL_FONT)
        self.label.pack(side=tk.RIGHT)

        directory = ["authorize_straight", "", "authorize_right", "authorize_left"]
        command = ["BLINK FEW TIMES", "LOOK RIGHT", "LOOK LEFT"]

        for i in range(len(directory)):
            if (directory[i] != ""):
                if os.path.exists(directory[i]):
                    shutil.rmtree(directory[i])
                os.makedirs(directory[i])

                for j in range(15):
                    self.controller.after(j * 1000, self.take_photo, directory[i], j)

            if (i + 1) < len(directory):
                self.controller.after((i + 1) * 5000, self.change_label, command[i])

        # root.after(5000, change_label, "BLINK FEW TIMES")

        self.controller.after(20000, self.change_label, "AUTHORIZATION", True, button_load_state, button_save_state)

    def load(self):
        print("Load chosen!")
        self.button_load['state'] = tk.DISABLED
        self.button_authorize['state'] = tk.NORMAL

    def save(self):
        print("Save chosen!")
        self.button_load['state'] = tk.NORMAL
        self.button_save['state'] = tk.DISABLED
        directory = "train_model"
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

        #np.savetxt(directory + '/straight_pca.model', straight_pca)
        #np.savetxt(directory + '/straight_sift.model', straight_sift)
        #np.savetxt(directory + '/left_pca.model', left_pca)
        #np.savetxt(directory + '/left_sift.model', left_sift)
        #np.savetxt(directory + '/right_pca.model', right_pca)
        #np.savetxt(directory + '/right_sift.model', right_sift)

    def show_detections(self, head_pose_detections, frame):
        for det in head_pose_detections:
            cv2.rectangle(frame, (det.x_min, det.y_min), (det.x_max, det.y_max), (0, 255, 0), 1)
            cv2.imshow('cropped', det.cropped_img)

    def show_video(self):
        if not self.cap.isOpened():
            print("ERROR: cannot open the camera")

        flag, img_new = self.cap.read()

        if flag is None:
            print("ERROR: cannot read the camera!")
        elif flag:
            self.last_img = img_new.copy()
            self.cur_img = cv2.cvtColor(self.last_img, cv2.COLOR_BGR2RGB)
            self.cur_img = cv2.resize(self.cur_img, (180, 120))

            head_pose_detections = self.controller.DHP.detect(self.cur_img)
            self.show_detections(head_pose_detections, self.cur_img)

            img = Image.fromarray(self.cur_img)

            imgtk = ImageTk.PhotoImage(image=img)
            self.lmain.imgtk = imgtk
            self.lmain.configure(image=imgtk)
            self.lmain.after(100, self.show_video)


def main():
    app = FaceApp()
    app.geometry("600x380+300+300")
    app.mainloop()


if __name__ == '__main__':
    main()

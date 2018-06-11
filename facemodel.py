import os
import Queue
import threading
import time
import numpy as np
import cv2
import utils
import os
import shutil
from gazedetector import *

from sklearn import svm
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class FaceModel():
    def __init__(self):
        self.in_cam_img_queue = Queue.Queue()
        self.outqueue = Queue.Queue()
        self.res_lock = threading.Lock()

        self.cam_img = None
        self.res_img = None

        self.is_trained = False
        self.is_training = False
        self.training_thread = None
        self.training_info = ""

        self.is_authorized = False
        self.is_authorizing = False
        self.authorizing_thread = None
        self.authorizing_info = ""

        self.GazeD = GazeDetector("./etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf",
                                  "./etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf",
                                  "./etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf")

        self.lfw_dataset = datasets.fetch_lfw_people(min_faces_per_person=50)
        l, h, w = self.lfw_dataset.images.shape
        self.training_data_dim = [w, h]
        print("number of training images: " + str(l))
        print("WxH: " + str(w) + "x" + str(h) + "=" + str(w * h) + " pixels")
        print("number of persons: " + str(len(self.lfw_dataset.target_names)))

        self.trained_ids = dict()


    def set_cam_image(self, img):
        # only add new data if available: triggers working thread!
        if self.in_cam_img_queue.empty():
            self.in_cam_img_queue.put(img.copy(), block=False)


    def get_res_image(self):
        # allow polling of result img
        self.res_lock.acquire()
        img = None
        try:
            img = self.res_img
        finally:
            self.res_lock.release()
        return img


    def get_info(self):
        if self.is_training:
            return self.training_info

        if self.is_authorizing:
            return self.authorizing_info

        txt = "no info..."
        if self.is_trained:
            txt = "model is trained"
        if self.is_authorized:
            txt = "authorized"
        return txt


    def train_model(self):
        if not self.is_training:
            self.training_thread = threading.Thread(target=self.train_model_thread).start()
        else:
            print("is already training!")


    def train_model_thread(self, num_image_per_side = 15, save_images=True):
        self.is_training = True

        states = ["straight", "left", "right"]
        state_angles = { "straight" : [-10, 10], "left" : [-90, -35], "right" : [35, 90] }
        img_buffer = { "straight" : [], "left" : [], "right" : [] }

        while self.is_training:
            # check if new image is available:

            img_buffer = self.capture_images(states, state_angles, num_image_per_side)


            self.training_info = "model training ..."
            if save_images:
                self.store_images(states, img_buffer)


            self.model_training(["straight"], img_buffer)
            self.training_info = "model training DONE!"
            self.is_training = False

        self.is_trained = True
        print("worker is done...")


    def capture_images(self, states, state_angles, num_image_per_side = 15):
        img_buffer = dict()
        for state in states:
            cnt = 0

            img_buffer[state] = []
            while cnt < num_image_per_side:
                if not self.in_cam_img_queue.empty():
                    cam_img = self.in_cam_img_queue.get(block=False)
                    print("do training")
                    # time.sleep(0.5)

                    head_pose_detections = self.detect_head_poses(cam_img)
                    self.show_detections(head_pose_detections, cam_img)

                    self.res_lock.acquire()
                    try:
                        self.res_img = cam_img
                        cv2.rectangle(self.res_img, (2, 2), (20, 20), (0, 255, 0), 2)
                    finally:
                        self.res_lock.release()

                    min_angle, max_angle = state_angles[state]
                    cur_angle = 0
                    img_cropped = None
                    if len(head_pose_detections):
                        det = head_pose_detections[0]
                        cur_angle = det.yaw
                        img_cropped = det.cropped_img
                    self.training_info = state + ": " + str(cnt) + "; expected angles: " + str(min_angle) + "," + str(
                        max_angle) + "\n cur angle: " + str(cur_angle)

                    if cur_angle > min_angle and cur_angle < max_angle and img_cropped is not None:
                        cnt += 1
                        img_buffer[state].append(img_cropped)

        return img_buffer


    def model_training(self, states, img_buffer):
        # do the model training
        X = self.lfw_dataset.data.copy()
        y = self.lfw_dataset.target.copy()
        target_names = self.lfw_dataset.target_names.copy()

        print("prev. X_len " + str(len(X)) + "; prev. y_len " + str(len(y)))
        y_len = len(y)
        t = 0
        for state in states:
            target_names = np.append(target_names, state)

            self.trained_ids[str(state)] = y_len + t

            for img in img_buffer[state]:
                resized_img = cv2.resize(img, (self.training_data_dim[0], self.training_data_dim[1]))
                np_face = np.asarray(resized_img.flatten(), dtype=np.float32)
                np_face /= 255.0  # scale uint8 coded colors from 0.0->1.0
                np_face = np_face.reshape((1, len(np_face)))  ## make a row vector

                X = np.append(X, np_face, axis=0)  ## append to the image vector
                y = np.append(y, y_len + t)  ## append to the label vector
            t += 1

        # shuffle data:
        indices = np.arange(len(y))
        np.random.RandomState(42).shuffle(indices)
        X, y = X[indices], y[indices]

        print("X_len " + str(len(X)) + "; y_len " + str(len(y)) + "; num targets: " + str(len(target_names)))

        # X = preprocessing.scale(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # apply pca and find eigenvectors and eigenvalues
        feature_dim = 200
        pca = PCA(n_components=feature_dim, whiten=True).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        c = svm.SVC()

        c.fit(X_train_pca, y_train)
        y_pred = c.predict(X_test_pca)
        print(classification_report(y_test, y_pred, target_names=target_names))
        i = 10
        print("Predicted:", target_names[y_pred[i]], " - Correct:", target_names[y_test[i]])


    def store_images(self, states, img_buffer):
        cwd = os.getcwd()
        print("saving images: " + str(cwd))

        for state in states:
            i = 0
            directory = "training_images/" + state
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

            for img in img_buffer[state]:
                cv2.imwrite(directory + "/" + str(i) + ".jpg", img)
                i += 1


    def stop_thread(self):
        if self.is_training:
            self.is_training = False
            self.training_thread.join()
        if self.is_authorizing:
            self.is_authorizing = False
            self.authorizing_thread.join()


    def show_detections(self, head_pose_detections, frame):
        for det in head_pose_detections:
            utils.draw_axis(frame, det.yaw, det.pitch, det.roll, tdx=(det.x_min + det.x_max) / 2,
                      tdy=(det.y_min + det.y_max) / 2, size=det.bbox_height / 2)
            cv2.rectangle(frame, (det.x_min, det.y_min), (det.x_max, det.y_max), (0, 255, 0), 1)


    def detect_head_poses(self, cam_img):
        return self.GazeD.detect(cam_img)


    def save_model(self, filename):
        print("saving model: " + filename)
        return False


    def load_model(self, filename):
        print("loading model: " + filename)
        return False

    def authorize(self):
        if not self.is_authorizing and self.is_trained:
            self.authorizing_thread = threading.Thread(target=self.authorize_thread).start()
        else:
            print("is already authorizing!")

    def authorize_thread(self):
        self.is_authorizing = True

        while(self.is_authorizing):
            # llooooooop
            print("looping...")
            time.sleep(0.2)
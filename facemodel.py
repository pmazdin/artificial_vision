import os
import Queue
import threading
import time
import numpy as np
import cv2
import utils
import os
import shutil
import dlib
import pickle
from gazedetector import *
from sift import *

import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils import face_utils

import svmdetector
import frdetector
import siftdetector

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

        self.blink_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # dlib's face detector

        self.SVM = svmdetector.SVMDetector()
        self.SIFT = siftdetector.SIFTDetector()
        self.FR = frdetector.FRDetector()



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
            #print("1")
            return self.training_info

        if self.is_authorizing:
            #print("2")
            return self.authorizing_info

        txt = "Wait for info"
        if self.is_trained:
            #print("3")
            txt = "Model is trained"
        if self.is_authorized:
            #print("4")
            txt = "Authorized"
        return txt


    def train_model(self):
        if not self.is_training:
            self.training_thread = threading.Thread(target=self.train_model_thread).start()
        else:
            print("is already training!")


    def train_model_thread(self, num_image_per_side = 10, save_images=False, load_images=True):
        self.is_training = True

        states = ["straight", "left", "right"]
        state_angles = { "straight" : [-10, 10], "left" : [-90, -35], "right" : [35, 90] }

        while self.is_training:
            # check if new image is available:

            if load_images:
                img_buffer = self.load_images(states, num_image_per_side)
            else:
                img_buffer = self.capture_images(states, state_angles, num_image_per_side)

            self.training_info = "IN PROGRESS"
            if save_images and not load_images:
                self.store_images(states, img_buffer)


            self.model_training(["straight"], img_buffer)
            self.training_info = "Training DONE!"
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
                        img_cropped = det.cropped_clr_img
                    self.training_info = "LOOK " +  state + ": " + str(cnt) + "/" + str(num_image_per_side) #+  "; expected angles: " + str(min_angle) + "," + str( max_angle) + "\n cur angle: " + str(cur_angle)

                    if cur_angle > min_angle and cur_angle < max_angle and img_cropped is not None:
                        cnt += 1
                        img_buffer[state].append(img_cropped)

        return img_buffer


    def model_training(self, states, img_buffer):
        self.SVM.training(img_buffer, states)

        train_buffer = dict()
        for state in states:
            train_buffer[state] = img_buffer[state][0]  # just take the first picture!

        self.SIFT.training(train_buffer, states)
        self.FR.training(train_buffer, states)
        self.is_trained = True


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

    def load_images(self, states, num_image_per_side = 15):
        cwd = os.getcwd()
        print("loading images from: " + str(cwd))
        img_buffer = dict()
        for state in states:
            directory = "training_images/" + state
            img_buffer[state] = []
            if not os.path.exists(directory):
                print("Error path does not exist!: " + directory)
            else:
                for i in range(num_image_per_side):
                    img_buffer[state].append(cv2.imread(directory + "/" + str(i) + ".jpg"))
                    i += 1

        return img_buffer

    def store_training_data(self, X):
        cwd = os.getcwd()
        print("saving training images: " + str(cwd))
        directory = "training_images/db"
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
        for i in range(len(X)):
            cv_img_train = self.get_cv_face(X[i])
            cv2.imwrite(directory + "/db_" + str(i) + ".jpg", cv_img_train)

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
        # print("saving model: " + filename)
        try:
            pickle.dump(self.classifier, open(filename, 'wb'))
            return True
        except Exception:
            return False


    def load_model(self, filename):
        # print("loading model: " + filename)

        try:
            self.classifier = pickle.load(open(filename, 'rb'))
            return True
        except Exception:
            return False

    def authorize(self):
        if not self.is_authorizing and self.is_trained:
            self.authorizing_thread = threading.Thread(target=self.authorize_thread).start()
        else:
            print("is already authorizing!")



    def calculate_ratio(self, eye):

        vert_one = dist.euclidean(eye[1], eye[5])  # vertical eye landmarks
        vert_two = dist.euclidean(eye[2], eye[4])
        horizontal = dist.euclidean(eye[0], eye[3])  # horizontal eye landmark
        ratio = (vert_one + vert_two) / (2.0 * horizontal)

        return ratio


    def detect_blinking(self, cam_img, face, RATIO_THRESHOLD):
        self.authorizing_info = "BLINK SLOWLY"
        (left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]  # load features
        (right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        shape = self.blink_predictor(cam_img, face)
        shape = face_utils.shape_to_np(shape)  # convert the landmark to np array

        left_eye = shape[left_start:left_end]  # left eye coordinates
        left_ratio = self.calculate_ratio(left_eye)

        right_eye = shape[right_start:right_end]  # right eye coordinates
        right_ratio = self.calculate_ratio(right_eye)

        total_ratio = (left_ratio + right_ratio) / 2.0  # avg ratio

        if total_ratio < RATIO_THRESHOLD:
            return True


    def authorize_thread(self):
        self.is_authorizing = True

        RATIO_THRESHOLD = 0.3  # blink detection
        blinking_test_done = False

        while(self.is_authorizing):

            if not self.in_cam_img_queue.empty():
                cam_img = self.in_cam_img_queue.get(block=False)
                print("looping...")
                head_pose_detections = self.detect_head_poses(cam_img)
                self.show_detections(head_pose_detections, cam_img)

                self.authorizing_info = "authorizing..."
                self.res_lock.acquire()
                try:
                    self.res_img = cam_img
                    cv2.rectangle(self.res_img, (2, 2), (20, 20), (255, 0, 0), 2)
                finally:
                    self.res_lock.release()

                if len(head_pose_detections):
                    det = head_pose_detections[0]
                    img_cropped = det.cropped_clr_img.copy()

                    if self.is_trained:

                        if not blinking_test_done:
                            self.authorizing_info = "BLINK SLOWLY"
                            face = dlib.rectangle(det.x_min, det.y_min, det.x_max, det.y_max)
                            if self.detect_blinking(cam_img, face, RATIO_THRESHOLD):
                                print("Blink detected")
                                blinking_test_done = True

                        if blinking_test_done:
                            USE_SIFT = False
                            USE_RT = True
                            USE_SVM = False

                            self.authorizing_info = "FAILED!"
                            print(self.authorizing_info)
                            if USE_SIFT:
                                matches = self.SIFT.compare(img_cropped, 15, 0.6)
                                self.authorizing_info = str(matches)
                            if USE_RT:
                                matches = self.FR.compare(img_cropped)
                                self.authorizing_info = str(matches)
                            if USE_SVM:
                                matches = self.SVM.compare(img_cropped)
                                self.authorizing_info = str(matches)



            #time.sleep(0.2)


        print("authorize thread done...")
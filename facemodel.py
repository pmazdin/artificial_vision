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

from sklearn import svm
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils import face_utils

import face_recognition

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
        self.classifier = None
        self.pca = None
        self.target_names = []

        self.SIFT_models = dict()
        self.SIFT_detector = cv2.xfeatures2d.SIFT_create()

        self.fr_encodings = dict()
        self.fr_names = []

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


    def train_model_thread(self, num_image_per_side = 10, save_images=False, load_images=False):
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
                    self.training_info = "LOOK " +  state + ": " + str(cnt) + "; expected angles: " + str(min_angle) + "," + str( max_angle) + "\n cur angle: " + str(cur_angle)

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
                np_face = self.get_np_face(img)
                X = np.append(X, np_face, axis=0)  ## append to the image vector
                y = np.append(y, y_len + t)  ## append to the label vector
            t += 1

        # shuffle data:
        indices = np.arange(len(y))
        np.random.RandomState(42).shuffle(indices)
        X_old = X.copy()
        X, y = X[indices], y[indices]

        print("X_len " + str(len(X)) + "; y_len " + str(len(y)) + "; num targets: " + str(len(target_names)))

        # X = preprocessing.scale(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # apply pca and find eigenvectors and eigenvalues
        feature_dim = 200
        self.pca = PCA(n_components=feature_dim, whiten=True).fit(X_train)
        X_train_pca = self.pca.transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        self.classifier = svm.SVC()

        self.classifier.fit(X_train_pca, y_train)
        y_pred = self.classifier.predict(X_test_pca)
        print(classification_report(y_test, y_pred, target_names=target_names))
        i = 10
        if y_test[i] < len(target_names):
          print("Predicted:", target_names[y_pred[i]], " - Correct:", target_names[y_test[i]])

        self.store_training_data(X_train)
        #for i in range(10):
        #    cwd = os.getcwd()
        #    cv_img_train = self.get_cv_face(X_train[i])
        #    cv2.imwrite(cwd + "/db_" + str(i) + ".jpg", cv_img_train)
        #for i in range(10):
        #    cv2.imwrite(cwd + "/tr_" + str(i) + ".jpg", self.get_cv_face(X_old[len(X_old)-1-i]))

        self.target_names = target_names.copy()

        # extract sift features

        self.extract_SIFT(states, img_buffer, self.SIFT_models)

        self.extract_FR(states, img_buffer)
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


    def extract_SIFT(self, states, img_buffer, SIFT_models):
        print("extracting SIFT models...")
        for state in states:
            SIFT_models[state] = []
            for img in img_buffer[state]:
                kp1, des1 = self.SIFT_detector.detectAndCompute(img, None)
                SIFT_models[state].append((kp1, des1))

    def extract_FR(self, states, img_buffer):
        self.fr_names = states
        for state in states:
            img = img_buffer[state][0]
            self.fr_encodings[state] = face_recognition.face_encodings(img)


    def detect_FR(self, states, img):
        faces_names = []
        for state in states:
            face_encodings = face_recognition.face_encodings(img)
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)

                for state in states:
                    matches = face_recognition.compare_faces(self.fr_encodings[state], face_encoding)
                    name = "Unknown"

                    # If a match was found in known_face_encodings, just use the first one.
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.fr_names[first_match_index]
                        faces_names.append(name)

        return faces_names

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

    def get_np_face(self, img):
        [w, h] = self.training_data_dim
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized_img = cv2.resize(img, (w, h))
        np_face = utils.cv_image_to_numpyarray(resized_img)
        return np_face

    def calculate_ratio(self, eye):

        vert_one = dist.euclidean(eye[1], eye[5])  # vertical eye landmarks
        vert_two = dist.euclidean(eye[2], eye[4])
        horizontal = dist.euclidean(eye[0], eye[3])  # horizontal eye landmark
        ratio = (vert_one + vert_two) / (2.0 * horizontal)

        return ratio

    def get_cv_face(self, np_face):
        [w, h] = self.training_data_dim
        img = utils.numpyarray_image_to_cv(np_face, w, h)
        return img


    def authorize_thread(self):
        self.is_authorizing = True

        RATIO_THRESHOLD = 0.3  # blink detection
        NB_FRAMES = 3  # number of frames under threshold
        REQUIRED_NB_BLINKS = 0  # number of detected blinks required

        success_cnt = 0

        cnt = 0  # frame counter
        total_nb = 0  # total number of detected blinks

        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # dlib's face detector

        (left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]  # load features
        (right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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
                        [w, h] = self.training_data_dim
                        np_face = self.get_np_face(img_cropped)

                        faces = np.zeros((1, h * w), dtype=np.float32)
                        faces[0, :] = np_face
                        cwd = os.getcwd()
                        cv2.imwrite(cwd + "/test_"  + ".jpg", self.get_cv_face(np_face))
                        X_test_pca = self.pca.transform(faces)
                        y_pred = self.classifier.predict(X_test_pca)

                        self.authorizing_info = "BLINK SLOWLY"

                        face = dlib.rectangle(det.x_min, det.y_min, det.x_max, det.y_max)

                        shape = predictor(cam_img, face)
                        shape = face_utils.shape_to_np(shape)  # convert the landmark to np array

                        left_eye = shape[left_start:left_end]  # left eye coordinates
                        left_ratio = self.calculate_ratio(left_eye)

                        right_eye = shape[right_start:right_end]  # right eye coordinates
                        right_ratio = self.calculate_ratio(right_eye)

                        total_ratio = (left_ratio + right_ratio) / 2.0  # avg ratio

                        if total_ratio < RATIO_THRESHOLD:
                            cnt += 1
                            print("Blink detected")
                        else:
                            if cnt >= NB_FRAMES:
                                total_nb += 1
                            cnt = 0  # reset the counter

                        if total_nb > REQUIRED_NB_BLINKS:
                            #print("Blink test done")
                            USE_SIFT = False
                            USE_RT = True
                            #print(y_pred.shape)
                            if USE_SIFT:
                                kp2, des2 = self.SIFT_detector.detectAndCompute(img_cropped, None)
                                [kp1, des1] = self.SIFT_models["straight"][10]
                                [num, ratio] = compare_ratio(des1,des2)
                                if num > 20:
                                     self.authorizing_info = str("SIFT RATIO: " + str(num) + "/" + str(ratio))

                                if(y_pred[0] < len(self.target_names)):
                                    self.authorizing_info = str("Predicted:" + self.target_names[y_pred[0]] + " - " + str(y_pred[0])) # + " - Correct:" + self.target_names[self.trained_ids["Straight"]])
                                    print(self.authorizing_info)
                                else:
                                    self.authorizing_info = str("Prediction failed..." + str(num) + "/" + str(ratio))
                                    print(self.authorizing_info)

                            if USE_RT:
                                face_names = self.detect_FR(["straight"], img_cropped)
                                if len(face_names):
                                    self.authorizing_info = "SUCCESS: detected: " + str(face_names) #str("SIFT SUCCESS: " + "{0:.2f}".format(ratio*100) + "%")
                                    print(self.authorizing_info)

                                    if success_cnt > 20:
                                        self.is_authorized = True
                                        self.is_authorizing = False
                                    success_cnt += 1
                                else:
                                    self.authorizing_info = "FAILED!"
                                    print(self.authorizing_info)
            #time.sleep(0.2)


        print("authorize thread done...")
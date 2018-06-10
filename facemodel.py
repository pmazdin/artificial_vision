import os
import Queue
import threading
import time
import cv2
import utils

from deep_head_pose.code import headposedlib
from gazedetector import *


class FaceModel():
    def __init__(self):
        self.in_cam_img_queue = Queue.Queue()
        self.outqueue = Queue.Queue()
        self.res_lock = threading.Lock()

        self.cam_img = None
        self.res_img = None

        self.is_trained = False
        self.is_training = False

        self.DHP = headposedlib.HeadPoseDLib("/home/jungr/workspace/NAV/development/face_authorization_py/deep_head_pose/hopenet_alpha2.pkl",
                                             "/home/jungr/workspace/NAV/development/face_authorization_py/deep_head_pose/mmod_human_face_detector.dat")

        self.GazeD = GazeDetector("/home/jungr/workspace/NAV/development/face_authorization_py/deepgaze/etc/xml/haarcascade_frontalface_alt.xml",
                                  "/home/jungr/workspace/NAV/development/face_authorization_py/deepgaze/etc/xml/haarcascade_profileface.xml")
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

    def train_model(self):
        if not self.is_training:
            threading.Thread(target=self.train_model_thread).start()
        else:
            print("is already training!")

    def train_model_thread(self):

        cnt = 0

        self.is_training = True
        while self.is_training and cnt < 1000:

            # check if new image is available:
            if not self.in_cam_img_queue.empty():
                cam_img = self.in_cam_img_queue.get(block=False)
                print("do training")
                #time.sleep(0.5)
                cnt += 1

                # TODO: DHP is synchronizing between main and worker thread!
                head_pose_detections = self.detect_head_poses(cam_img, "GAZE")
                self.show_detections(head_pose_detections, cam_img)
                self.res_lock.acquire()
                try:
                    self.res_img = cam_img
                    cv2.rectangle(self.res_img, (2,2), (20,20), (0, 255, 0), 2)
                finally:
                    self.res_lock.release()

        self.is_trained = True
        self.is_training = False
        print("worker is done...")




    def show_detections(self, head_pose_detections, frame):
        for det in head_pose_detections:
            utils.draw_axis(frame, det.yaw, det.pitch, det.roll, tdx=(det.x_min + det.x_max) / 2,
                      tdy=(det.y_min + det.y_max) / 2, size=det.bbox_height / 2)
            cv2.rectangle(frame, (det.x_min, det.y_min), (det.x_max, det.y_max), (0, 255, 0), 1)


    def detect_head_poses(self, cam_img, method="DHP"):
        if method == "DHP":
            print("using DHP")
            return self.DHP.detect(cam_img)
        elif method == "GAZE":
            print("using GAZE")
            return self.GazeD.detect2(cam_img)
        else:
            print(method + " not supported!")

        return None
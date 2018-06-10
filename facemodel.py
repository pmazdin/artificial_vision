import os
import Queue
import threading
import time
import cv2
import utils

from deep_head_pose.code import headposedlib
from deepgaze.deepgaze import *
from deepgaze.deepgaze.face_detection import HaarFaceDetector
from gazecnnhpe import *

class GazeDetector:
    def __init__(self, frontal_xml="./etc/xml/haarcascade_frontalface_alt.xml", profile_xml="./etc/xml/haarcascade_profileface.xml"):
        self.hfd = HaarFaceDetector(frontal_xml, profile_xml )

        self.hpe = GazeCnnHPE("/home/jungr/workspace/NAV/development/face_authorization_py/deepgaze/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf",
                              "/home/jungr/workspace/NAV/development/face_authorization_py/deepgaze/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf",
                              "/home/jungr/workspace/NAV/development/face_authorization_py/deepgaze/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf")

    def detect(self, image):
        clr_img = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        allTheFaces = self.hfd.returnMultipleFacesPosition(image, runFrontal=True, runFrontalRotated=True,
                                                      runLeft=True, runRight=True,
                                                      frontalScaleFactor=1.2, rotatedFrontalScaleFactor=1.2,
                                                      leftScaleFactor=1.2, rightScaleFactor=1.2,
                                                      minSizeX=64, minSizeY=64,
                                                      rotationAngleCCW=30, rotationAngleCW=-30)
        head_pose_detections = []
        # Iterating all the faces
        for element in allTheFaces:
            x_min = int(element[0])
            y_min = int(element[1])
            x_max = int(x_min+ element[2])
            y_max = int(y_min + element[3])
            # Drawing a rectangle around the face
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), [255, 0, 0])
            img_crop = image[y_min:y_max, x_min:x_max]
            res = headposedlib.HeadPoseDetection()
            res.confidence = 1

            crop_clr_img = clr_img[y_min:y_max, x_min:x_max]
            cv2.resize(crop_clr_img, (64, 64))
            [res.roll, res.pitch, res.yaw] = self.hpe.detect(crop_clr_img)
            res.cropped_img = img_crop.copy()
            res.x_min = x_min
            res.y_min = y_min
            res.x_max = x_max
            res.y_max = y_max
            res.bbox_height = element[3]
            res.bbox_width = element[2]
            head_pose_detections.append(res)

        return head_pose_detections


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

                # TODO: is synchronizing between main and worker thread!
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
            return self.GazeD.detect(cam_img)
        else:
            print(method + " not supported!")

        return None

import os
import Queue
import threading
import time
import cv2
import utils

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
        self.training_thread = None
        self.training_info = ""

        self.GazeD = GazeDetector("./etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf",
                                  "./etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf",
                                  "./etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf")
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

        if self.is_trained:
            return "model is trained"

        return "no info..."

    def train_model(self):
        if not self.is_training:
            self.training_thread = threading.Thread(target=self.train_model_thread).start()
        else:
            print("is already training!")

    def train_model_thread(self, num_image_per_side = 15, save_images=False):
        self.is_training = True

        states = ["straight", "left", "right"]
        state_angles = { "straight" : [-10, 10], "left" : [-90, -35], "right" : [35, 90] }
        img_buffer = { "straight" : [], "left" : [], "right" : [] }


        while self.is_training:
            # check if new image is available:


            for state in states :
                cnt = 0

                while cnt < num_image_per_side:
                    if not self.in_cam_img_queue.empty():
                        cam_img = self.in_cam_img_queue.get(block=False)
                        print("do training")
                        #time.sleep(0.5)


                        head_pose_detections = self.detect_head_poses(cam_img)
                        self.show_detections(head_pose_detections, cam_img)

                        self.res_lock.acquire()
                        try:
                            self.res_img = cam_img
                            cv2.rectangle(self.res_img, (2,2), (20,20), (0, 255, 0), 2)
                        finally:
                            self.res_lock.release()


                        min_angle, max_angle = state_angles[state]
                        cur_angle = 0
                        img_cropped = None
                        if len(head_pose_detections):
                            det = head_pose_detections[0]
                            cur_angle = det.yaw
                            img_cropped = det.cropped_img
                        self.training_info = state + ": " + str(cnt) + "; expected angles: " + str(min_angle) + "," + str(max_angle) + "\n cur angle: " + str(cur_angle)

                        if cur_angle > min_angle and cur_angle < max_angle and img_cropped is not None:
                            cnt += 1
                            img_buffer[state].append(img_cropped)

            self.is_training = False

            if save_images:
                print("saving iamges: ")



            # do the model training



        self.is_trained = True
        print("worker is done...")

    def stop_thread(self):
        self.is_training = False
        self.training_thread.join()

    def show_detections(self, head_pose_detections, frame):
        for det in head_pose_detections:
            utils.draw_axis(frame, det.yaw, det.pitch, det.roll, tdx=(det.x_min + det.x_max) / 2,
                      tdy=(det.y_min + det.y_max) / 2, size=det.bbox_height / 2)
            cv2.rectangle(frame, (det.x_min, det.y_min), (det.x_max, det.y_max), (0, 255, 0), 1)


    def detect_head_poses(self, cam_img):
        return self.GazeD.detect(cam_img)


import cv2
import utils

from headposedetection import HeadPoseDetection
from gazecnnhpe import *
from dlibfacedetector import *


class GazeDetector:
    def __init__(self, roll_file="../../etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf",
                 pitch_file="../../etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf",
                 yaw_file="../../etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"):

        self.hpe = GazeCnnHPE(roll_file, pitch_file, yaw_file)
        self.facedetect = DLibFaceDetector()


    def detect(self, image):
        clr_img = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        allTheFaces = self.facedetect.detect(image)
        head_pose_detections = []
        # Iterating all the faces
        for element in allTheFaces:
            x_min = int(element[0])
            y_min = int(element[1])
            x_max = int(element[2])
            y_max = int(element[3])
            # Drawing a rectangle around the face
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), [255, 0, 0])
            img_crop = image[y_min:y_max, x_min:x_max]
            res = HeadPoseDetection()
            res.confidence = 1

            crop_clr_img = clr_img[y_min:y_max, x_min:x_max]
            # print("bb: " + str(x_min) + "," + str(x_max) + ";" + str(y_min) + "," + str(y_max))
            crop_clr_img = cv2.resize(crop_clr_img, (128, 128))
            # print(str(crop_clr_img.shape))

            [res.roll, res.pitch, res.yaw] = self.hpe.detect(crop_clr_img)
            res.cropped_img = img_crop.copy()
            res.cropped_clr_img =  crop_clr_img.copy()
            res.x_min = x_min
            res.y_min = y_min
            res.x_max = x_max
            res.y_max = y_max
            res.bbox_height = y_max - y_min
            res.bbox_width = x_max - x_min
            head_pose_detections.append(res)

        return head_pose_detections

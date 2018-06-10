import cv2
import utils

from deep_head_pose.code import headposedlib
from deepgaze.deepgaze.face_detection import HaarFaceDetector
from gazecnnhpe import *
from dlibfacedetector import *


class GazeDetector:
    def __init__(self, frontal_xml="./etc/xml/haarcascade_frontalface_alt.xml",
                 profile_xml="./etc/xml/haarcascade_profileface.xml"):
        self.hfd = HaarFaceDetector(frontal_xml, profile_xml)

        self.hpe = GazeCnnHPE(
            "/home/jungr/workspace/NAV/development/face_authorization_py/deepgaze/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf",
            "/home/jungr/workspace/NAV/development/face_authorization_py/deepgaze/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf",
            "/home/jungr/workspace/NAV/development/face_authorization_py/deepgaze/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf")

        self.facedetect = DLibFaceDetector()

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
            x_max = int(x_min + element[2])
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

    def detect2(self, image):
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
            res = headposedlib.HeadPoseDetection()
            res.confidence = 1

            crop_clr_img = clr_img[y_min:y_max, x_min:x_max]
            # print("bb: " + str(x_min) + "," + str(x_max) + ";" + str(y_min) + "," + str(y_max))
            crop_clr_img = cv2.resize(crop_clr_img, (128, 128))
            # print(str(crop_clr_img.shape))

            [res.roll, res.pitch, res.yaw] = self.hpe.detect(crop_clr_img)
            res.cropped_img = img_crop.copy()
            res.x_min = x_min
            res.y_min = y_min
            res.x_max = x_max
            res.y_max = y_max
            res.bbox_height = y_max - y_min
            res.bbox_width = x_max - x_min
            head_pose_detections.append(res)

        return head_pose_detections
import os
import tensorflow as tf
import cv2
from deepgaze.deepgaze.head_pose_estimation import CnnHeadPoseEstimator


class GazeCnnHPE:
    def __init__(self, roll_file="../../etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf",
                 pitch_file="../../etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf",
                 yaw_file="../../etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"):
        self.sess = tf.Session() #Launch the graph in a session.
        self.HPE = CnnHeadPoseEstimator(self.sess) #Head pose estimation object

        # Load the weights from the configuration folders
        self.HPE.load_roll_variables(os.path.realpath(roll_file))
        self.HPE.load_pitch_variables(os.path.realpath(pitch_file))
        self.HPE.load_yaw_variables(os.path.realpath(yaw_file))


    def detect(self, image):
        # Get the angles for roll, pitch and yaw
        roll = self.HPE.return_roll(image)  # Evaluate the roll angle using a CNN
        pitch = self.HPE.return_pitch(image)  # Evaluate the pitch angle using a CNN
        yaw = self.HPE.return_yaw(image)  # Evaluate the yaw angle using a CNN
        print("Estimated [roll, pitch, yaw] ..... [" + str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")

        return [roll, pitch, yaw]




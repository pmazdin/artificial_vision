import numpy as np
import cv2
import face_recognition
import os

class FRDetector:
    def __init__(self):
        self.encodings = dict()
        self.train_img = dict()
        self.names = []

    def train(self, img, name):
        self.encodings[name] = face_recognition.face_encodings(img)
        self.train_img[name] = img.copy()
        if name not in self.names:
            self.names.append(name)


    def training(self, img_buffer, names):
        for name in names:
            if img_buffer.has_key(name):
                if len(img_buffer[name]):
                    self.train(img_buffer[name], name)

    def compare(self, img):
        matches = []
        for name in self.names:
            face_encodings = face_recognition.face_encodings(img)
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)

                for n in self.names:
                    matches = face_recognition.compare_faces(self.encodings[n], face_encoding)
                    found_name = "Unknown"

                    # If a match was found in known_face_encodings, just use the first one.
                    if True in matches:
                        first_match_index = matches.index(True)
                        matches.append(self.names[first_match_index])
        return matches


def test_FRDetector():
    dir = os.getcwd() + "/training_images/"
    train_idx = 2
    types = ["straight", "left", "right"]

    d = FRDetector()

    for t in types:
        d.train(cv2.imread(dir + t + "/" + str(train_idx) + ".jpg"), t)  # trainImage

    good_cnt = 0
    cnt = 0

    for i in range(20):
        for t in types:
            matches = d.compare(cv2.imread(dir + t + "/" + str(i) + ".jpg"))  # queryImage

            cnt += 1
            if t in matches:
                good_cnt += 1

    print("matching ratio: " + str(good_cnt/(1.0*cnt)))



if __name__ == '__main__':
    test_FRDetector()
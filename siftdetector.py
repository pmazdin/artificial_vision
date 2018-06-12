import cv2
import sift
import os

class SIFTDetector:
    def __init__(self):
        self.detector = cv2.xfeatures2d.SIFT_create()
        self.encodings = dict()
        self.train_img = dict()
        self.names = []

    def train(self, img, name):
        kp1, des1 = self.detector.detectAndCompute(img, None)
        self.encodings[name] = [kp1, des1]
        self.train_img[name] = img.copy()
        if name not in self.names:
            self.names.append(name)

    def training(self, img_buffer, names):
        for name in names:
            if img_buffer.has_key(name):
                if len(img_buffer[name]):
                    self.train(img_buffer[name], name)

    def compare(self, img, min_num_good = 25, ratio_thres = 0.75, draw_matches= False):
        matches = []
        [kp1, des1] = self.detector.detectAndCompute(img, None)
        for name in self.names:
            [kp_n, des_n] = self.encodings[name]

            [num_good, ratio, matches_] = sift.compare_ratio(des1, des_n)
            if num_good > min_num_good and ratio > ratio_thres:
                matches.append(name)

            if draw_matches:
                img_res = sift.draw_matches(img, kp1, self.train_img[name], kp_n, matches_)
                cv2.imshow('Matched Features', img_res)
                cv2.waitKey(0)

        return matches


def test_SIFTDetector():
    dir = os.getcwd() + "/training_images/"
    train_idx = 2
    types = ["straight", "left", "right"]

    d = SIFTDetector()

    for t in types:
        d.train(cv2.imread(dir + t + "/" + str(train_idx) + ".jpg"), t)  # trainImage

    good_cnt = 0
    cnt = 0

    for i in range(20):
        for t in types:
            matches = d.compare(cv2.imread(dir + t + "/" + str(i) + ".jpg"), draw_matches=False)  # queryImage

            cnt += 1
            if t in matches:
                if len(matches) == 1:
                    good_cnt += 1

    print("matching ratio: " + str(good_cnt/(1.0*cnt)))



if __name__ == '__main__':
    test_SIFTDetector()
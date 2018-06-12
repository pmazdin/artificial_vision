import numpy as np
import cv2
import os
import utils
import shutil
from sklearn import svm
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA


class SVMDetector:
    def __init__(self, min_faces_per_person=50):

        self.lfw_dataset = datasets.fetch_lfw_people(min_faces_per_person=min_faces_per_person)
        self.l, self.h, self.w = self.lfw_dataset.images.shape
        self.training_data_dim = [self.w, self.h]
        print("number of training images: " + str(self.l))
        print("WxH: " + str(self.w) + "x" + str(self.h) + "=" + str(self.w * self.h) + " pixels")
        print("number of persons: " + str(len(self.lfw_dataset.target_names)))


        self.classifier = None
        self.pca = None
        self.target_names = []
        self.trained_ids = dict()
        self.is_trained = False

    def training(self, img_buffer, names):
        X = self.lfw_dataset.data.copy()
        y = self.lfw_dataset.target.copy()
        target_names = self.lfw_dataset.target_names.copy()

        print("prev. X_len " + str(len(X)) + "; prev. y_len " + str(len(y)))
        y_len = len(y)
        t = 0

        for name in names:
            target_names = np.append(target_names, name)

            self.trained_ids[str(name)] = y_len + t

            for img in img_buffer[name]:
                np_face = self.get_np_face(img)
                X = np.append(X, np_face, axis=0)  ## append to the image vector
                y = np.append(y, y_len + t)  ## append to the label vector
            t += 1

        # shuffle data:
        indices = np.arange(len(y))
        np.random.RandomState(42).shuffle(indices)
        X, y = X[indices], y[indices]

        print("X_len " + str(len(X)) + "; y_len " + str(len(y)) + "; num targets: " + str(len(target_names)))

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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

        self.target_names = target_names.copy()
        self.is_trained = True

    def compare(self, img):
        matches = []
        if self.is_trained:
            [w, h] = self.training_data_dim
            np_face = self.get_np_face(img)

            # PCA + SVM:
            faces = np.zeros((1, h * w), dtype=np.float32)
            faces[0, :] = np_face
            cwd = os.getcwd()
            cv2.imwrite(cwd + "/test_" + ".jpg", self.get_cv_face(np_face))
            X_test_pca = self.pca.transform(faces)
            y_pred = self.classifier.predict(X_test_pca)

            id = y_pred[0]
            if id < len(self.target_names):
                print("found: " + self.target_names[id])
                matches.append(self.target_names[id])

        return matches

    def get_np_face(self, img):
        [w, h] = self.training_data_dim
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized_img = cv2.resize(img, (w, h))
        np_face = utils.cv_image_to_numpyarray(resized_img)
        return np_face

    def get_cv_face(self, np_face):
        [w, h] = self.training_data_dim
        img = utils.numpyarray_image_to_cv(np_face, w, h)
        return img


import dlib


class DLibFaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        faces_array = self.detector(image, 1)
        print("Total Faces: " + str(len(faces_array)))

        ROIs = []
        for i, pos in enumerate(faces_array):
            min_x = pos.left()
            min_y = pos.top()
            max_x = pos.right()
            max_y = pos.bottom()
            ROIs.append([min_x, min_y, max_x, max_y])

        return ROIs
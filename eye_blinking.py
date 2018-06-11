from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import cv2

RATIO_THRESHOLD = 0.3 #blink detection
NB_FRAMES = 3 #number of frames under threshold
REQUIRED_NB_BLINKS = 5 #number of detected blinks required


def calculate_ratio(eye):

    vert_one = dist.euclidean(eye[1], eye[5]) #vertical eye landmarks
    vert_two = dist.euclidean(eye[2], eye[4])
    horizontal = dist.euclidean(eye[0], eye[3]) #horizontal eye landmark
    ratio = (vert_one + vert_two) / (2.0 * horizontal)

    return ratio

cnt = 0 #frame counter
total_nb = 0 # total number of detected blinks

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #dlib's face detector

(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] #load features
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

video_stream = cv2.VideoCapture(0)

while total_nb < REQUIRED_NB_BLINKS:

    flag, frame = video_stream.read()
    if not flag:
        break

    frame = cv2.resize(frame, (450, 300)) #resize the frame
    face_dcts = detector(frame, 0) #detected faces

    for face in face_dcts:
        shape = predictor(frame, face)
        shape = face_utils.shape_to_np(shape) #convert the landmark to np array

        left_eye = shape[left_start:left_end] #left eye coordinates
        left_ratio = calculate_ratio(left_eye)

        right_eye = shape[right_start:right_end]  # right eye coordinates
        right_ratio = calculate_ratio(right_eye)

        total_ratio = (left_ratio + right_ratio) / 2.0 #avg ratio

        if total_ratio < RATIO_THRESHOLD:
            cnt += 1
        else:
            if cnt >= NB_FRAMES:
                total_nb += 1
            cnt = 0 #reset the counter

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
cv2.destroyAllWindows()


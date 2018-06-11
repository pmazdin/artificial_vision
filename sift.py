import cv2
import numpy as np


def drawMatches(img1, kp1, img2, kp2, matches, knn=False):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    print(len(matches))
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1] = np.dstack([img1])
    out[:rows2, cols1:] = np.dstack([img2])
    for mat in matches:
        if knn:
            img1_idx = mat[0].queryIdx
            img2_idx = mat[0].trainIdx
        else:
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0, 1), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0, 1), 1)
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0, 1), 1)

    return out


def get_good_matches(matches, ratio = 0.8):
    good_matches = []
    for mat in matches:

        if mat[0].distance < ratio * mat[1].distance:
            good_matches.append(mat)

    return mat


def symetry_test(matches_L, matches_R):
    good_matches = []

    for L in matches_L:
        for R in matches_R:
            if L.queryIdx == R.trainIdx and R.queryIdx == L.trainIdx:
                good_matches.append(L)
                break

    return good_matches

def compare_ratio(des1, des2):
    bf = cv2.BFMatcher()
    matches_LR = bf.match(des1, des2)
    matches_RL = bf.match(des2, des1)
    good_matches = symetry_test(matches_LR, matches_RL)

    return [len(good_matches) , len(good_matches)/(0.5*(len(des1) + len(des2)))]


def compare(filename1, filename2, knn=False):
    img1 = cv2.imread(filename1)  # queryImage
    img2 = cv2.imread(filename2)  # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    if knn:
        matches = bf.knnMatch(des1, des2, 2)
        good_matches = get_good_matches(matches)
        img3 = drawMatches(img1, kp1, img2, kp2, matches, knn=True)
        img4 = drawMatches(img1, kp1, img2, kp2, good_matches)
    else:
        matches = bf.match(des1, des2)
        matches_RL = bf.match(des2, des1)
        good_matches = symetry_test(matches, matches_RL)
        img3 = drawMatches(img1, kp1, img2, kp2, matches)
        img4 = drawMatches(img1, kp1, img2, kp2, good_matches)

        [num, ratio] = compare_ratio(des1, des2)
        print("ratio: " + str(num) + "/" + str(ratio))


    # Show the image
    cv2.imshow('Matched Features', img3)
    cv2.waitKey(0)
    cv2.imshow('Good Matched Features', img4)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')
    return len(matches)

def test_sift():
    dir = "training_images/straight/"

    for i in range(20):
        file_one = dir+ str(i) + ".jpg"
        file_two = dir+ str(i+5) + ".jpg"
        print(compare(file_one, file_two))


if __name__ == '__main__':
    test_sift()

# print(compare(file_two, file_one))
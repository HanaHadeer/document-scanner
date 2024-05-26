import cv2
from cv2 import imshow, waitKey, VideoCapture
import numpy as np
from datetime import datetime

cap = VideoCapture(1)  # value (1) for a second camera device (phone), (0) for webcam


# Rescale cam window if you use phone cam, to show the img in a small window
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


# Convert img to gray -> blur -> canny
def processing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)

    # For edge detection thickness
    kernel = np.ones((5, 5))  # Matrix of ones (5 by 5)
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)

    # return imgGray
    # return imgBlur
    # return imgCanny
    return imgThres


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 9000:  # 9000 or more than 10000 for high resolution cameras
            # cv2.drawContours(imgContour, cnt, -1, (0, 0, 255), 3)
            peri = cv2.arcLength(cnt, True)  # True bcs the paper is a closed shape (rectangle)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # approx return number of paper_angles (must be 4)

            if area > maxArea and len(approx) == 4:
                biggest = approx  # retrieve the biggest shape in the pic, which is the paper
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (0, 0, 255), 3)
    return biggest


def reorder(points):
    points = points.reshape((4, 2))
    newPoints = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]
    return newPoints


def warp(img, biggest, imsSize):
    widthImg = imgSize[0]
    heightImg = imsSize[1]
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOutput[20: imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]  # Crop
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))  # Resize

    return imgCropped


while True:
    _, img = cap.read()
    imgSize = img.shape
    imgContour = img.copy()  # Used in getContours() to make changes on img_copy
    processedImg = processing(img)

    biggest = getContours(processedImg)

    print(getContours(processedImg))  # print the points

    if biggest.size != 0:
        imgWarped = warp(img, biggest, imgSize)
        key = cv2.waitKey(1)

        if key == ord('s'):
            # cv2.imwrite('myDoc.jpg', imgWarped)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'myDoc_{timestamp}.jpg'
            cv2.imwrite(filename, imgWarped)

        rescaleImgWarped = rescale_frame(imgWarped, percent=50)
        cv2.imshow('Document', rescaleImgWarped)
    else:
        pass

    # if you use the phone's camera, call rescale_frame() to reduce the window size
    # rescale_img = rescale_frame(img, percent=50)
    rescale_img = rescale_frame(imgContour, percent=50)
    imshow('imgContour', rescale_img)
    rescale_processedImg = rescale_frame(processedImg, percent=50)
    imshow('processedImg', rescale_processedImg)

    # imshow('img', img)    # test
    # imshow('imgContour', imgContour)
    # imshow('processedImg',processedImg)

    # Create a wait key, when the user clicks (q) the window will close
    if waitKey(1) & 0xFF == ord('q'):
        break

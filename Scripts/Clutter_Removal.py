import cv2
import numpy as np

def clutter_removal(img):
    kernelSizes = [(1, 1), (3, 3), (5, 5), (9, 9)]
    for i in kernelSizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, i)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow("Opening: ({}, {})".format(i[0], i[1]), opening)
    return opening
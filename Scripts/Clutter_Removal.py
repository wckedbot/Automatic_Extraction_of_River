import cv2
import numpy as np

def clutter_removal(img):
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, se2)

    mask = np.dstack([mask, mask, mask]) / 255
    return mask
import cv2
import numpy as np
from matplotlib import pyplot as plt

def hist_thres(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    histg= plt.hist(img.ravel(),256,[0,256])
    print(histg)
    start = np.min(np.where(histg[0] > 0))
    end = np.max(np.where(histg[0] > 0))
    center = (start + end) // 2
    left_weight = np.sum(histg[0][0:center+1])
    right_weight = np.sum(histg[0][center+1:end+1])
    print(start, end, center, left_weight, right_weight)
    while start != end:
        if right_weight > left_weight:
            right_weight -= histg[0][end]
            end -= 1
            if ((start + end)//2) < center:
                left_weight -= histg[0][center]
                right_weight += histg[0][center]
                center -= 1
        else:
            left_weight -= histg[0][start]
            start += 1
            if ((start + end)//2) >= center:
                left_weight += histg[0][center+1]
                right_weight -= histg[0][center+1]
                center += 1
    print(center)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] > center:
                img[i][j] = 0
            else:
                img[i][j] = 255
    return img

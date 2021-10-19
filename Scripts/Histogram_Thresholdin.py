import cv2
import numpy as np

def hist_thres(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img)
    # cv2.imshow("binary", img)
    start = np.min(np.where(img[0]>0))
    end = np.max(np.where(img[0]>0))
    print(img[0][end])
    print(end)
    center = (start + end) // 2
    print(center)
    left_weight = np.sum(img[0][0:center+1])
    right_weight = np.sum(img[0][center+1:end+1])
    print(left_weight, right_weight)
    while(start != end):
        if right_weight > left_weight:
            right_weight -= img[0][end]
            end -= 1
            if ((start + end)//2) < center:
                left_weight -= img[0][center]
                right_weight += img[0][center]
                center -= 1
        else:
            left_weight -= img[0][start]
            start += 1
            if ((start + end)//2) >= center:
                left_weight += img[0][center+1]
                right_weight -= img[0][center+1]
                center += 1
        print(left_weight, right_weight)
        print(start, end, center)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] > 100:
                img[i][j] = 0
            else:
                img[i][j] = 255
    return img

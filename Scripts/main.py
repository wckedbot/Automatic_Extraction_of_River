import cv2

from Decorrelation_Stretch import *
from Histogram_Thresholdin import *
from matplotlib import pyplot as plt
from Clutter_Removal import *
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = cv2.imread(r"./Input_Image/1.jpg")
    img = img[3069:5267, 3000:4800]
    img = cv2.resize(img, (480, 480))
    cv2.imshow('image', img)
    decorr_img = decorrstretch(img, 0)
    cv2.imshow("decorr", decorr_img)
    # cv2.imwrite('../Output_Image', decorr_img)
    seg_img = hist_thres(decorr_img)
    cv2.imshow("segmented image", seg_img)
    final_img = clutter_removal(seg_img)
    # cv2.imshow("Noise Free", final_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

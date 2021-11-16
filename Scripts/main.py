import cv2
from matplotlib import colors
from Decorrelation_Stretch import *
from Histogram_Thresholdin import *
from matplotlib import pyplot as plt
from Clutter_Removal import *
from Outline import *
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = cv2.imread(r"./Input_Image/1.png")
    cv2.imshow('image', img)
    cv2.imwrite('1.png', img)

    figure, axis = plt.subplots(1, 1)
    axis.hist(img.ravel(), 256, [0, 256])
    r, g, b = cv2.split(img)
    fig1 = plt.figure()
    axis1 = fig1.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis1.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis1.set_xlabel("Red")
    axis1.set_ylabel("Green")
    axis1.set_zlabel("Blue")

    decorr_img = decorrstretch(img, 0)
    figure2, axis2 = plt.subplots(1, 1)
    axis2.hist(decorr_img.ravel(), 256, [0, 256])
    cv2.imshow("decorr", decorr_img)
    r, g, b = cv2.split(decorr_img)
    fig3 = plt.figure()
    axis3 = fig3.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = decorr_img.reshape((np.shape(decorr_img)[0] * np.shape(decorr_img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis3.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis3.set_xlabel("Red")
    axis3.set_ylabel("Green")
    axis3.set_zlabel("Blue")


    # cv2.imwrite('../Output_Image', decorr_img)
    seg_img = hist_thres(decorr_img)
    cv2.imshow("segmented image", seg_img)
    seg_img = clutter_removal(seg_img)
    cv2.imshow("Noise Free", seg_img)
    output = Outline(seg_img)
    cv2.imshow("Outlined Image", output)


    plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

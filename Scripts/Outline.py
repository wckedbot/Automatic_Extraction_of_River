import cv2
import numpy as np
from matplotlib import pyplot as plt

def Outline(seg_img):
    seg_img = seg_img.astype(np.uint8)
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('sd',seg_img)
    main = cv2.imread('./Input_Image/1.png', cv2.IMREAD_GRAYSCALE)
    main = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)


    RGBforLabel = {1: (0, 0, 255), 2: (0, 255, 255)}

    # Find external contours
    contours, _ = cv2.findContours(seg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Iterate over all contours
    for i, c in enumerate(contours):
        # Find mean colour inside this contour by doing a masked mean
        mask = np.zeros(seg_img.shape, np.uint8)
        cv2.drawContours(mask, [c.astype(int)], -1, 255, -1)
        # DEBUG: cv2.imwrite(f"mask-{i}.png",mask)
        mean, _, _, _ = cv2.mean(seg_img, mask=mask)
        # DEBUG: print(f"i: {i}, mean: {mean}")

        # Get appropriate colour for this label
        label = 2 if mean > 1.0 else 1
        colour = RGBforLabel.get(label)
        # DEBUG: print(f"Colour: {colour}")

        # Outline contour in that colour on main image, line thickness=1
        cv2.drawContours(main, [c], -1, colour, 1)
    return main
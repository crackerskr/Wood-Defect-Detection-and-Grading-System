import numpy as np
import cv2
from matplotlib import pyplot as plt


def isUndersized(s):
    img = cv2.imread(s)
    # cv2.imshow("Original", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray", gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    histEqual = clahe.apply(gray)
    # cv2.imshow("Hist", histEqual)

    ret, th = cv2.threshold(histEqual, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # cv2.imshow("th", th)

    gaussianBlur =  cv2.GaussianBlur(th, (3,3), 0)

    #----------------------------------------------------------------------------------------------------------------
    kernel = np.ones((5,5),np.uint8)
    img_open = cv2.morphologyEx(gaussianBlur, cv2.MORPH_OPEN, kernel)
    img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3,3),np.uint8)
    dilated = cv2.dilate(img_close, kernel)
    # cv2.imshow("dilated", dilated)
    # plt.imshow(dilated)
    # plt.show()
    #----------------------------------------------------------------------------------------------------------------

    cnts, hierachy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clone = img.copy()
    largest_area = 0
    for (i, c) in enumerate(cnts):
        # calculate the area and perimeter
        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        if area > largest_area:
            largest_area = area
            largest_c = c
        # print('shape #{} -- Area: {}  ,  perimeter: {}'.format(str(i+1), area, perimeter))
        approximation =  cv2.approxPolyDP(c, 0.01*perimeter, True)

    (x,y,w,h) =  cv2.boundingRect(largest_c)
    mask = mask = np.zeros(gray.shape, np.uint8)
    mask[x:x+w, y:y+h] = 255
    masked_img = cv2.bitwise_and(gray,gray,mask = mask)

    # compute gradients along the x and y axis, respectively
    gX = cv2.Sobel(masked_img, cv2.CV_64F, 1, 0)
    gY = cv2.Sobel(masked_img, cv2.CV_64F, 0, 1)
    # compute the gradient magnitude and orientation
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))

    largest_magnitude = 0
    for i in magnitude:
        for j in i:
            if j > largest_magnitude:
                largest_magnitude = j

    # cv2.rectangle(clone, (x,y), (x+w, y+h), (0,255,0), 10)
    # cv2.putText(clone, "w={},h={}".format(w,h), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)

    # cv2.imshow('Rect Contours', clone)
    # clone = clone[:, :, [2, 1, 0]]
    # plt.imshow(clone)
    # plt.show()

    # cv2.waitKey(0)
    # cv2.destroyAllWindows

    if(largest_magnitude>900):
        return True
    else:
        return False
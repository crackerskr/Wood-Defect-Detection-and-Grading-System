# importing necessary libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt

# read a cracked sample image
def hasCrack(s):

    img = cv2.imread(s)

    # Convert into gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    r1 = 250
    def pixel_value(img, r1):
        if (img>r1):
            return 180
        else:
            return img
    picture_value = np.vectorize(pixel_value)

    pw = picture_value(gray, r1)
    pw = pw.astype(np.uint8)


    # Apply logarithmic transform
    img_log = (np.log(pw+1)/(np.log(1+np.max(pw))))*255

    # Specify the data type
    img_log = np.array(img_log,dtype=np.uint8)
    # plt.imshow(img_log,cmap="gray")
    # plt.show()

    blur = cv2.GaussianBlur(img_log,(3,3),0)
    # cv2.imshow("Blur", pw)

    # ret, thresh1 = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY_INV)
    # plt.imshow(thresh1, cmap='gray')
    # plt.show()

    # Canny Edge Detection
    edges = cv2.Canny(blur,150,210)
    # plt.imshow(edges,cmap="gray")
    # plt.show()

    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    dilated = cv2.dilate(opening, (5,5))

    cnts, hierachy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    clone = img.copy()
    for (i, c) in enumerate(cnts):
        # calculate the area and perimeter
        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        # print('shape #{} -- Area: {}  ,  perimeter: {}'.format(str(i+1), area, perimeter))
        approximation =  cv2.approxPolyDP(c, 0.01*perimeter, True)
        (x,y,w,h) =  cv2.boundingRect(c)
        cv2.rectangle(clone, (x,y), (x+w, y+h), (0,255,0), 10)


    # rgb_img = img[:, :, [2, 1, 0]]
    # plt.subplot(211),plt.imshow(rgb_img)
    # plt.title('Original'),plt.xticks([]), plt.yticks([])
    # featuredImg = clone[:, :, [2, 1, 0]]
    # plt.subplot(212),plt.imshow(featuredImg)
    # plt.title('Output Image'),plt.xticks([]), plt.yticks([])
    # plt.show()



    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if(len(cnts) >=5):
        return True
    else:
        return False
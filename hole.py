import numpy as np
import cv2
import matplotlib.pyplot as plt

# read a cracked sample image
def getNumHoles(s):
    img = cv2.imread(s)

    # Convert into gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',gray)

    blur = cv2.blur(gray,(3,3))
    #cv2.imshow('blur',blur)

    # Apply logarithmic transform
    img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255
    # Specify the data type
    img_log = np.array(img_log,dtype=np.uint8)
    #cv2.imshow('log',img_log)

    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(img_log, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("Closing", closing)

    # Create feature detecting method
    orb = cv2.ORB_create(nfeatures=1000)
    kp, descriptors = orb.detectAndCompute(closing, None)

    zeros = np.uint8(np.zeros(img.shape))
    featuredImg = (cv2.drawKeypoints(zeros, kp, None, color=(255,255,255)))
    featuredImg = cv2.cvtColor(featuredImg, cv2.COLOR_BGR2GRAY)
    # plt.imshow(featuredImg, cmap='gray')
    # plt.show()
    ret, th = cv2.threshold(featuredImg, 100, 255, cv2.THRESH_BINARY)
    # plt.imshow(th, cmap='gray')
    # plt.show()


    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, (5,5))
    dilate = cv2.dilate(closing, (3,3))

    cnts, hierachy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    numHoles = 0
    clone = img.copy()
    for (i, c) in enumerate(cnts):
        # calculate the area and perimeter
        perimeter =  cv2.arcLength(c, True)
        approximations = cv2.approxPolyDP(c, 0.01*perimeter, True ) 
        area = cv2.contourArea(c)
        # To find the center
        M = cv2.moments(c)
        cx =  int(M['m10']/M['m00'])
        cy =  int(M['m01']/M['m00'])
        if area>50:
            ((cx,cy), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(clone, (int(cx),int(cy)), int(radius), (0,255,0), 5)
            # print('shape #{} -- Area: {}  ,  perimeter: {}'.format(str(i+1), area, perimeter))
            numHoles = numHoles + 1


    # cv2.imshow('Original', img)
    # cv2.imshow('Output', clone)

    # rgb_img= img[:,:, [2,1,0]]
    # clone = clone[:, :, [2, 1, 0]]
    # plt.subplot(211)
    # plt.imshow(rgb_img, cmap='hot', interpolation='bicubic') # read image as RGB
    # plt.title('Original'),plt.xticks([]), plt.yticks([])
    # plt.subplot(212)
    # featuredImg = clone[:, :, [2, 1, 0]]
    # plt.imshow(clone, interpolation='bicubic')
    # plt.title('Output Image'),plt.xticks([]), plt.yticks([])
    # plt.show()

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return numHoles
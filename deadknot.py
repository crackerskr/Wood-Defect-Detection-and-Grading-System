import numpy as np
import cv2
import matplotlib.pyplot as plt

def hasDeadKnot(s):
    # Read image with greyscale
    image = cv2.imread(s)
    # plt.imshow(image)
    # plt.show()

    # Convert into gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray)
    # plt.show()

    # Smoothing averagely
    blurred_image = cv2.GaussianBlur(gray,(5,5),0)
    # plt.imshow(blurred_image)
    # plt.show()

    # Apply Threshold
    ret, thresh = cv2.threshold(blurred_image, 20, 255, cv2.THRESH_BINARY_INV)
    # plt.imshow(thresh)
    # plt.show()

    kernel = np.ones((7,7), np.uint8)
    image_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # plt.imshow(image_open)
    # plt.show()
    image_close = cv2.morphologyEx(image_open, cv2.MORPH_CLOSE, kernel)
    # plt.imshow(image_close)
    # plt.show()
    dilated = cv2.dilate(image_close,kernel)
    # plt.imshow(dilated)
    # plt.show()

    #Find contours of dead knots
    countours, hierachy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clone = image.copy()
    hasKnot = False
        
    for (i, c) in enumerate(countours):
        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)    
        approximation =  cv2.approxPolyDP(c, 0.01*perimeter, True)
        # # To find the center
        M = cv2.moments(c)
        cx =  int(M['m10']/M['m00'])
        cy =  int(M['m01']/M['m00'])
        if area>1000: #To remove some noise
            ((cx,cy), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(clone, (int(cx),int(cy)), int(radius), (255,0,0), 5)
            # print('shape #{} -- Area: {}  ,  perimeter: {}'.format(str(i+1), area, perimeter))
            hasKnot = True

    # Final Output for comparison of original image and detected image
    # rgb_img = image[:, :, [2, 1, 0]] #convert bmp to rgb image
    # plt.subplot(211)
    # plt.imshow(rgb_img, cmap='hot', interpolation='bicubic') # read image as RGB
    # plt.title('Original'),plt.xticks([]), plt.yticks([])
    # plt.subplot(212)
    # featuredImg = clone[:, :, [2, 1, 0]]
    # plt.imshow(featuredImg, interpolation='bicubic')
    # plt.title('Output Image'),plt.xticks([]), plt.yticks([])
    # plt.show()

    return hasKnot
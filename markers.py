#!/usr/bin/env python
import cv2
import numpy as np
import time

cv2.namedWindow("preview")

markdict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
for x in range(4):
    outimg = cv2.aruco.drawMarker(markdict, x, 200) #, outimg)
    cv2.imshow("preview", outimg)
    # cv2.imwrite('marker'+str(x)+'.png', outimg)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

markerSize = 93
outimg = cv2.aruco.drawMarker(markdict, 5, markerSize) 
height = 1050
width = 1680

while True:

    bigPic  = np.zeros((height,width))
    yo = np.random.randint(0,width-markerSize)
    xo = np.random.randint(0,height-markerSize)
    bigPic[xo:xo+markerSize,yo:yo+markerSize] = outimg
    cv2.imshow("preview", bigPic) 
    # time.sleep(3) 
    
    cv2.waitKey(-1)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break


cv2.destroyWindow("preview")

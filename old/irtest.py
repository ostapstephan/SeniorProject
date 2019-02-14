#import lib.pbcvt as pbcvt

import cv2
import numpy as np
import sys
from time import time
'''
def distance(o1, o2):
    (x1,y1,w1,h1) = o1
    (x2,y2,w2,h2) = o2
    c1 = (x1+w1/2,y1+h1/2)
    c2 = (x2+w2/2,y2+h2/2)
    return np.hypot(c1[0]-c2[0],c1[1]-c2[1])

def label_corners(pointList):
    if 2==len(pointlist): 
        pts = {}
        if pointList[0][1] < pointList[1][1]:
            pts('top') = pointList[0]
            pts('bottom') = pointList[1]
        else: 
            pts('top') = pointList[1]
            pts('bottom') = pointList[0]
        return pts, 2

    elif len(pointlist) ==4:
        for i in range(4):
            pointList[i][0] # x coord of all 
            #TODO

    else:
        return "Wrong point val. Supplied value = " + str(len(pointlist))
'''

cv2.namedWindow("preview")
vc = cv2.VideoCapture(-1)
print(vc.get(3))
print(vc.get(4))
# vout = None
# if (int(sys.argv[5])):
#     fourcc = cv2.VideoWriter_fourcc(*'x264')
#     vout = cv2.VideoWriter('pupiltest.mp4', fourcc, 24.0, (int(vc.get(3)),int(vc.get(4))))

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255
# Filter by Area.
params.filterByArea = True
params.minArea = 50
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.4
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.4
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.3
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

ptime = time()
nf = 0
while rval:
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = 255-roi_gray
    # thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours, -1, (0,0,255), 3)
    # for cont in contours:
    #     if len(cont) > 5:# and cv2.contourArea(cont) > 1000:
    #         ellipse = cv2.fitEllipse(cont)
    #         cv2.ellipse(frame, ellipse, (0,0,255),2)
    #         cv2.circle(frame, (int(ellipse[0][0]),int(ellipse[0][1])), 2, (255,0,0), 3)

    keypoints = detector.detect(255-roi_gray)
    for point in keypoints:
        frame = cv2.drawMarker(frame, (int(point.pt[0]),int(point.pt[1])), (0,0,255))


    

    cv2.imshow("preview", frame)
    # if vout:
    #     vout.write(frame)
    nf = nf + 1
    if time() - ptime > 5:
        PointArray = []
        for pointt in keypoints:
            p = [int(pointt.pt[0]),int(pointt.pt[1])]
            PointArray.append(p)

        print(PointArray)



        print(str(nf/(time()-ptime)))
        ptime = time()
        nf = 0
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    elif key == 32:
        cv2.imwrite('testimage.png',frame);
    rval, frame = vc.read()

cv2.destroyWindow("preview")
vc.release()
# if vout:
#     vout.release()

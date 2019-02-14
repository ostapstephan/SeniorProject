#!/usr/bin/env python
import pbcvt
import cv2
import numpy as np
import sys
from time import time
from calibrate import calibrate

vc = cv2.VideoCapture(int(sys.argv[1]))
# vc = cv2.VideoCapture('http://199.98.27.252:6680/?action=stream')
print(vc.get(3))
print(vc.get(4))
# vout = None
# if (int(sys.argv[5])):
#     fourcc = cv2.VideoWriter_fourcc(*'x264')
#     vout = cv2.VideoWriter('pupiltest.mp4', fourcc, 24.0,
#                           (int(vc.get(3)), int(vc.get(4))))

roic = calibrate(vc)
calibrated = False
if roic[0] != roic[2] and roic[1] != roic[3]:
    calibrated = True

cv2.namedWindow("preview")
cv2.namedWindow("preview2")
ptime = time()
nf = 0
eye_cascade = cv2.CascadeClassifier('trained/haarcascade_eye.xml')
kernel = np.ones((5, 5), np.uint8)
thresh1 = 50
thresh2 = 11

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False
while rval:
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_color = frame

    if calibrated:
        cv2.rectangle(roi_color, roic[:2], roic[2:], (0, 0, 255), 2)

        eye_roi_gray = roi_gray[roic[1]:roic[3], roic[0]:roic[2]]

        eye_roi_gray = cv2.equalizeHist(eye_roi_gray)
        # eye_roi_gray = cv2.erode(eye_roi_gray,kernel)
        eye_roi_gray = cv2.GaussianBlur(eye_roi_gray, (25, 25), 0)

        roi_color[roic[1]:roic[3], roic[0]:roic[2], 0] = eye_roi_gray
        roi_color[roic[1]:roic[3], roic[0]:roic[2], 1] = eye_roi_gray
        roi_color[roic[1]:roic[3], roic[0]:roic[2], 2] = eye_roi_gray

        eye_roi_color = roi_color[roic[1]:roic[3], roic[0]:roic[2]]

    else:
        eye_roi_gray = roi_gray
        eye_roi_gray = cv2.equalizeHist(eye_roi_gray)
        eye_roi_gray = cv2.erode(eye_roi_gray, kernel)
        roi_color[:, :, 0] = eye_roi_gray
        roi_color[:, :, 1] = eye_roi_gray
        roi_color[:, :, 2] = eye_roi_gray
        eye_roi_color = roi_color

    thresh = None
    print(str(thresh1)+", "+str(thresh2))
    ret, thresh = cv2.threshold(eye_roi_gray, thresh1, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(eye_roi_gray, 255,
    #                                cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                cv2.THRESH_BINARY, 115, thresh1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(roi_color, contours, -1, (0,0,255), 3)
    for cont in contours:
        if len(cont) > 5 and 30000 > cv2.contourArea(cont) > 1000:
            conv = cv2.convexHull(cont)
            ellipse = cv2.fitEllipse(conv)
            cv2.ellipse(eye_roi_color, ellipse, (0, 0, 255), 2)
            cv2.circle(eye_roi_color, (int(ellipse[0][0]), int(ellipse[0][1])),
                       2, (255, 0, 0), 3)

    if thresh is not None:
        if calibrated:
            roi_gray[roic[1]:roic[3], roic[0]:roic[2]] = thresh
        else:
            roi_gray = thresh

    cv2.imshow("preview", roi_color)
    cv2.imshow("preview2", roi_gray)
    # if vout:
    #     vout.write(frame)
    nf = nf + 1
    if time() - ptime > 5:
        print(str(nf/(time()-ptime)))
        ptime = time()
        nf = 0
        eyes = None
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
    elif key == 32:
        cv2.imwrite('testimage.png', frame)
    elif key == 104:
        thresh1 = thresh1 + 5
    elif key == 106:
        thresh1 = thresh1 - 5
    elif key == 107:
        thresh1 = thresh1 + 1
    elif key == 108:
        thresh1 = thresh1 - 1
    rval, frame = vc.read()

cv2.destroyWindow("preview")
cv2.destroyWindow("preview2")
vc.release()
# if vout:
#     vout.release()

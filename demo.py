#!/usr/bin/env python
import pbcvt
import cv2
# import sys
from time import time
from calibrate import calibrate


def draw_ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness=3, lineType=cv2.LINE_AA, shift=10):
    center = (
        int(round(center[0] * 2**shift)),
        int(round(center[1] * 2**shift))
    )
    axes = (
        int(round(axes[0] * 2**shift)),
        int(round(axes[1] * 2**shift))
    )
    cv2.ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness, lineType, shift)


# vc = cv2.VideoCapture(int(sys.argv[1]))
vc = cv2.VideoCapture('http://raspberrypi0:8080/?action=stream')
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
ptime = time()
nf = 0

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
        # eye_roi_gray = cv2.GaussianBlur(eye_roi_gray, (25, 25), 0)

        roi_color[roic[1]:roic[3], roic[0]:roic[2], 0] = eye_roi_gray
        roi_color[roic[1]:roic[3], roic[0]:roic[2], 1] = eye_roi_gray
        roi_color[roic[1]:roic[3], roic[0]:roic[2], 2] = eye_roi_gray

        eye_roi_color = roi_color[roic[1]:roic[3], roic[0]:roic[2]]

    else:
        eye_roi_gray = roi_gray
        eye_roi_gray = cv2.equalizeHist(eye_roi_gray)
        # eye_roi_gray = cv2.GaussianBlur(eye_roi_gray, (25, 25), 0)

        roi_color[:, :, 0] = eye_roi_gray
        roi_color[:, :, 1] = eye_roi_gray
        roi_color[:, :, 2] = eye_roi_gray

        eye_roi_color = roi_color

    out = pbcvt.findPupilEllipse(eye_roi_gray)
    draw_ellipse(eye_roi_color, (out[0], out[1]), (out[2], out[3]), out[4],
                 0, 360, (0, 255, 0), 2)

    cv2.imshow("preview", roi_color)
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
    rval, frame = vc.read()

cv2.destroyWindow("preview")
vc.release()
# if vout:
#     vout.release()

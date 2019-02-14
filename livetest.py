#!/usr/bin/env python
import pbcvt
import cv2
import sys
from time import time


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


cap = cv2.VideoCapture(int(sys.argv[1]))
cv2.namedWindow('test')
while(True):
    ret, frame = cap.read()

    t = time()
    out = pbcvt.findPupilEllipse(frame)
    print(time()-t)
    draw_ellipse(frame, (out[0], out[1]), (out[2]/2, out[3]/2), out[4],
                 0, 360, (0, 0, 0), 2)

    cv2.imshow('test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

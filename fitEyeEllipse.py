#!/usr/bin/env python
import pbcvt
print("survived the import")
import cv2
import math
import numpy as np
from time import time
from time import sleep
from params import pupil_tracker_params

TIMEOUT = 10000


def draw_ellipse(
        img,
        center,
        axes,
        angle,
        startAngle,
        endAngle,
        color,
        thickness=3,
        lineType=cv2.LINE_AA,
        shift=10
):
    center = (int(round(center[0] * 2**shift)), int(round(center[1] * 2**shift)))
    axes = (int(round(axes[0] * 2**shift)), int(round(axes[1] * 2**shift)))
    cv2.ellipse(
        img,
        center,
        axes,
        angle,
        startAngle,
        endAngle,
        color,
        thickness,
        lineType,
        shift
    )


def putEllipse(queue, image):
    queue.put(pbcvt.findPupilEllipse(image, TIMEOUT))


with open('data/p1-right/pupil-ellipses.txt') as f:
    s = f.read().split('\n')

s = {
    int(x.split('|')[0]): [float(y) for y in x.split('|')[1:][0].split()]
    for x in s[:-1]
}

cv2.namedWindow('test')
xx = 0
t1 = time()
t0 = t1
totacc = 0
out = None
p = None
prev = None
while (True):
    start = time()
    # img = cv2.imread('data/simulated/render_eye_'+str(xx % 49)+'.png',
    #                  cv2.IMREAD_COLOR)
    img = cv2.imread(
        'data/p1-right/frames/' + str(xx % 939) + '-eye.png',
        cv2.IMREAD_COLOR
    )
    xx += 1

    out = pbcvt.findPupilEllipse(img, TIMEOUT, *pupil_tracker_params)
    draw_ellipse(img, (out[0], out[1]), (out[2], out[3]), out[4], 0, 360, (0, 0, 0), 2)

    if time() - start < (1 / 30):
        sleep((1 / 30) - (time() - start))

    cv2.imshow('test', img)

    if xx in s:
        diff = [(s[xx][d] - out[d])**2 for d in range(4)]
        diff += [(np.rad2deg(s[xx][4]) + 180 * (np.rad2deg(s[xx][4]) < 0) - out[4])**2]
        acc = math.sqrt(sum(diff))
        print('acc: ', acc)
        totacc += acc

    if xx % 10 == 0 and xx > 0:
        t2 = time()
        print('local: ', 10 / (t2 - t1))
        print('total: ', xx / (t2 - t0))
        t1 = t2

    if xx > 939:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('total acc: ', totacc / len(s))
cv2.destroyAllWindows()

#!/usr/bin/env python
import cv2
import sys
import numpy as np


def gkern(kernlen, sigma):
    # First a 1-D  Gaussian
    lim = kernlen // 2 + (kernlen % 2) / 2
    t = np.linspace(-lim, lim, kernlen)
    bump = np.exp(-0.25 * (t / sigma)**2)
    bump /= np.trapz(bump)  # normalize the integral to 1

    # make a 2-D kernel out of it
    return bump[:, np.newaxis] * bump[np.newaxis, :]


if len(sys.argv) == 2:
    img0 = cv2.imread(sys.argv[1])
    img0 = np.float64(img0 / 255)
    img1 = np.zeros(img0.shape)
    outfname = 'heatmapped-' + sys.argv[1]
elif len(sys.argv) == 3:
    img0 = np.zeros((int(sys.argv[1]), int(sys.argv[2]), 3))
    img1 = np.zeros((int(sys.argv[1]), int(sys.argv[2]), 3))
    outfname = 'heatmap_' + sys.argv[1] + 'x' + sys.argv[2] + '.png'
else:
    img0 = np.zeros((1050, 1680, 3))
    img1 = np.zeros((1050, 1680, 3))
    outfname = 'heatmap_1050x1680.png'

radius = 200
sigma = 30
gain = 500
decay = 1.007
mask = gkern(2 * radius + 1, sigma) * gain

cv2.namedWindow('preview')
curpos = [int(img1.shape[0] / 2), int(img1.shape[1] / 2)]

while True:

    xn = max(curpos[0] - radius, 0)
    yn = max(curpos[1] - radius, 0)
    xm = min(curpos[0] + radius + 1, img1.shape[0])
    ym = min(curpos[1] + radius + 1, img1.shape[1])
    kxn = radius - (curpos[0] - xn)
    kyn = radius - (curpos[1] - yn)
    kxm = radius + xm - curpos[0]
    kym = radius + ym - curpos[1]
    # print(curpos)
    # print((xn, yn), ' ', (xm, ym))
    # print((kxn, kyn), ' ', (kxm, kym))
    img1[xn:xm, yn:ym, 0] += mask[kxn:kxm, kyn:kym]
    img1[xn:xm, yn:ym, 1] -= mask[kxn:kxm, kyn:kym] / 4
    img1[xn:xm, yn:ym, 2] -= mask[kxn:kxm, kyn:kym] / 2
    img1[:, :, :] /= decay

    cv2.imshow('preview', img0 + img1)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('j'):
        curpos[0] += 5
    elif key & 0xFF == ord('k'):
        curpos[0] -= 5
    elif key & 0xFF == ord('l'):
        curpos[1] += 5
    elif key & 0xFF == ord('h'):
        curpos[1] -= 5

# cv2.imwrite(outfname, img0)
cv2.destroyAllWindows()

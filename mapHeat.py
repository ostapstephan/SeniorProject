#!/usr/bin/env python
import cv2
import sys
import numpy as np
import scipy.stats as st


def gkern(kernlen=21):
    """Returns a 2D Gaussian kernel."""

    lim = kernlen // 2 + (kernlen % 2) / 2
    x = np.linspace(-lim, lim, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


if len(sys.argv) == 2:
    img = cv2.imread(sys.argv[1])
    outfname = sys.argv[1] + '-heatmapped.png'
elif len(sys.argv) == 3:
    img = np.zeros((int(sys.argv[1]), int(sys.argv[2]), 3))
    outfname = 'heatmap_' + sys.argv[1] + 'x' + sys.argv[2] + '.png'
else:
    img = np.zeros((1050, 1680, 3))
    outfname = 'heatmap_1050x1680.png'

radius = 100
mask = gkern(2 * radius + 1) * 100

cv2.namedWindow('preview')
curpos = [int(img.shape[0] / 2), int(img.shape[1] / 2)]

while True:
    cv2.imshow('preview', img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('j'):
        curpos[0] += 1
    elif key & 0xFF == ord('k'):
        curpos[0] -= 1
    elif key & 0xFF == ord('l'):
        curpos[1] += 1
    elif key & 0xFF == ord('h'):
        curpos[1] -= 1
    elif key == 32:  # spacebar
        xn = max(curpos[0] - radius, 0)
        yn = max(curpos[1] - radius, 0)
        xm = min(curpos[0] + radius + 1, img.shape[0])
        ym = min(curpos[1] + radius + 1, img.shape[1])
        kxn = radius - (curpos[0] - xn)
        kyn = radius - (curpos[1] - yn)
        kxm = radius + xm - curpos[0] + 1
        kym = radius + ym - curpos[1] + 1
        print(curpos)
        print((xn, yn), ' ', (xm, ym))
        print((kxn, kyn), ' ', (kxm, kym))
        img[xn:xm, yn:ym, 0] += mask[kxn:kxm, kyn:kym]

cv2.imwrite(outfname, img)
cv2.destroyAllWindows()

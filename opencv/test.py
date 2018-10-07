import cv2
import numpy as np
import sys
from time import time

cv2.namedWindow("preview")
vc = cv2.VideoCapture(int(sys.argv[1]))
vc.set(3,1920)
vc.set(4,1080)
print(vc.get(3))
print(vc.get(4))

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

ptime = time()
nf = 0
while rval:
    cv2.imshow("preview", frame)
    nf = nf + 1
    if time() - ptime > 5:
        print(str(nf/(time()-ptime)))
        ptime = time()
        nf = 0
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    elif key == 32:
        cv2.imwrite('testimage.png',frame);

cv2.destroyWindow("preview")
vc.release()

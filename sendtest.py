import cv2
import numpy as np
from time import time

# cv2.namedWindow("preview2")
vt = cv2.VideoCapture('videotestsrc ! appsink', cv2.CAP_GSTREAMER)
vc = cv2.VideoWriter('appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ! queue !' \
                     'caps=video/x-raw,format=BGR,width=320,height=240,framerate=30/1 !' \
                     'videoconvert ! video/x-raw,format=I420 !' \
                     'x264enc speed-preset=ultrafast tune=zerolatency !' \
                     'rtph264pay config-interval=1 name=pay0 pt=96 !' \
                     'udpsink host=127.0.0.1 port=5000 sync=false' \
                     ,0,30.0,(320,240))

print(vt.get(3))
print(vt.get(4))

nf = 0
ptime = time()
while True:
    rval, frame = vt.read()

    if not rval:
        print('empty frame')
        continue

    # cv2.imshow("preview2", frame)
    vc.write(frame)

    nf += 1
    if time() - ptime > 5:
        print(str(nf/(time()-ptime)))
        ptime = time()
        nf = 0
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

# cv2.destroyWindow("preview2")
vt.release()
vc.release()

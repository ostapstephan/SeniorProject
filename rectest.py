import cv2
import numpy as np
from time import time

cv2.namedWindow("preview")
# vc = cv2.VideoCapture('udpsrc port=6666 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H265, payload=(int)96" ! rtph265depay ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
# vc = cv2.VideoCapture('udpsrc port=6666 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
# vc = cv2.VideoCapture('http://192.168.1.251:8080/?action=stream')
vc = cv2.VideoCapture('http://192.168.1.252:8080/?action=stream')
print(vc.get(3))
print(vc.get(4))

nf = 0
ptime = time()
while True:
    rval, frame = vc.read()

    if not rval:
        print('empty frame')
        continue

    cv2.imshow("preview", frame)
    nf += 1
    if time() - ptime > 5:
        print(str(nf/(time()-ptime)))
        ptime = time()
        nf = 0
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()

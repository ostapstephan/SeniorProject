#!/usr/bin/env python
import cv2
import sys

cap0 = cv2.VideoCapture(int(sys.argv[1]))
cap1 = cv2.VideoCapture(int(sys.argv[2]))
cv2.namedWindow('test0')
cv2.namedWindow('test1')
time = 0
while(True):
    ret, frame0 = cap0.read()
    ret, frame1 = cap1.read()

    cv2.imshow('test0', frame0)
    cv2.imshow('test1', frame1)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key == 32:  # spacebar
        cv2.imwrite('photos/0-'+str(time)+'.png', frame0)
        cv2.imwrite('photos/1-'+str(time)+'.png', frame1)
        time += 1

cap0.release()
cap1.release()
cv2.destroyAllWindows()

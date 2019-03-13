#!/usr/bin/env python
import cv2
import sys

cap0 = cv2.VideoCapture('https://raspberrypi0')
cap1 = cv2.VideoCapture('https://raspberrypi1')
cap2 = cv2.VideoCapture(int(sys.argv[1]))
cap3 = cv2.VideoCapture(int(sys.argv[2]))
cv2.namedWindow('test0')
cv2.namedWindow('test1')
cv2.namedWindow('test2')
cv2.namedWindow('test3')
time = 0
while(True):
    ret, frame0 = cap0.read()
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()
    ret, frame3 = cap3.read()

    cv2.imshow('test0', frame0)
    cv2.imshow('test1', frame1)
    cv2.imshow('test2', frame2)
    cv2.imshow('test3', frame3)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key == 32:  # spacebar
        cv2.imwrite('photos/0-'+str(time)+'.png', frame0)
        cv2.imwrite('photos/1-'+str(time)+'.png', frame1)
        cv2.imwrite('photos/2-'+str(time)+'.png', frame2)
        cv2.imwrite('photos/3-'+str(time)+'.png', frame3)
        time += 1

cap0.release()
cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()

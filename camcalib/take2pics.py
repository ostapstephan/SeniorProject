import cv2
import subprocess as sp
import sys
import numpy
import datetime

#simple script for the testing of homography with only 2 cameras 
#only uses 2 webcams

i=0
cap = [None,None]
j=0

while i < 2: 
    try:
        cap[i] = cv2.VideoCapture(int(j))
        ret, image = cap[i].read()
        if not ret :
            raise Exception
        print("accepted:",i,j)
        i+=1
    except:
        # print(i,j)
        pass
    j+=1
    if j > 20:
        break

# cap3.set(3,1280)
# cap3.set(4,720)

cv2.namedWindow('Video0')
cv2.namedWindow('Video1')
time = 0


while True:    
    _, image0 = cap[0].read()
    _, image1 = cap[1].read() 

    if image0 is not None:
        cv2.imshow('Video0', image0)
    if image1 is not None:
        cv2.imshow('Video1', image1)
 
    key = cv2.waitKey(1) 
    if key & 0xFF == ord('q'):
        break

    elif key == 32:  # spacebar
        cv2.imwrite('data/0-'+str(time)+'.png', image0)
        cv2.imwrite('data/1-'+str(time)+'.png', image1)
        time += 1

    # pipe0.stdout.flush()
    # pipe1.stdout.flush()

cap[0].release()
cap[1].release()

cv2.destroyAllWindows()

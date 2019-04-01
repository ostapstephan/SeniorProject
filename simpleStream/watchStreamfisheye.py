import cv2
import numpy as np
import sys
import time 

cap = cv2.VideoCapture("/dev/stdin")
ii=0
t1 = time.time()
while(True):
    # Capture frame-by-frame
    # for i in range(4):
        # cap.grab()
    
    ret, frame = cap.read()
    ii+=1 
    
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.(gray,1)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if ii%100 ==0: 
        print(100.0/(time.time()-t1))        
        t1 = time.time()
    # except:
        # print("meh")
# When everything done, release the capture

cap.release()
cv2.destroyAllWindows()

import lib.pbcvt as pbcvt

import cv2
import numpy as np
import sys
from time import time

def distance(o1, o2):
    (x1,y1,w1,h1) = o1
    (x2,y2,w2,h2) = o2
    c1 = (x1+w1/2,y1+h1/2)
    c2 = (x2+w2/2,y2+h2/2)
    return np.hypot(c1[0]-c2[0],c1[1]-c2[1])

cv2.namedWindow("preview")
cv2.namedWindow("preview2")
vc = cv2.VideoCapture(int(sys.argv[1]))
vc.set(3,int(sys.argv[2]))
vc.set(4,int(sys.argv[3]))
print(vc.get(3))
print(vc.get(4))
# vout = None
# if (int(sys.argv[5])):
#     fourcc = cv2.VideoWriter_fourcc(*'x264')
#     vout = cv2.VideoWriter('pupiltest.mp4', fourcc, 24.0, (int(vc.get(3)),int(vc.get(4))))

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

ptime = time()
nf = 0
# face_cascade = cv2.CascadeClassifier('trained/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('trained/haarcascade_eye.xml')
glass_cascade = cv2.CascadeClassifier('trained/haarcascade_eye_tree_eyeglasses.xml')
reye_cascade = cv2.CascadeClassifier('trained/haarcascade_righteye_2splits.xml')
leye_cascade = cv2.CascadeClassifier('trained/haarcascade_lefteye_2splits.xml')

# face = None
# flost = 0
while rval:
    roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_color = frame
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # flost = flost+1
    # for f in faces:
    #     if face is not None:
    #         # print("Face: " + str(distance(f,face)))
    #         if not (1 < distance(f,face) < 40):
    #             continue
    #     face = f
    #     flost = 0

    # if flost < 5 and face is not None:
    #     (x,y,w,h) = face
    #     x+=10
    #     y+=10
    #     w = int(w*0.85)
    #     h = int(h*0.5)
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #     roi_gray = gray[y:y+h, x:x+w]
    #     roi_color = frame[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for e in eyes:
        (ex,ey,ew,eh) = e
        # ex += 10
        # ey += 10
        # ew -= 10
        # eh -= 10
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
        eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
        eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]
        hist = cv2.calcHist([eye_roi_gray],[0],None,[256],[0,256])
        # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # Apply KMeans
        compactness,labels,centers = cv2.kmeans(hist,2,None,criteria,100,cv2.KMEANS_RANDOM_CENTERS)
        print(np.sqrt(compactness)/10)
        print(centers)
        # center = pbcvt.findPupil(roi_gray, int(ex), int(ey), int(ew), int(eh))
        ret, thresh = cv2.threshold(eye_roi_gray, centers[0]-10, 255, 0)
        # thresh = cv2.adaptiveThreshold(eye_roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(eye_roi_color, contours, -1, (0,0,255), 3)
        for cont in contours:
            if len(cont) > 5 and cv2.contourArea(cont) > 1000:
                ellipse = cv2.fitEllipse(cont)
                cv2.ellipse(eye_roi_color, ellipse, (0,0,255),2)
                cv2.circle(eye_roi_color, (int(ellipse[0][0]),int(ellipse[0][1])), 2, (255,0,0), 3)
        # cv2.circle(eye_roi_color, center, 2, (0,255,0), 3)

    # else:
    #     face = None


    cv2.imshow("preview", frame)
    cv2.imshow("preview2", thresh)
    # if vout:
    #     vout.write(frame)
    nf = nf + 1
    if time() - ptime > 5:
        print(str(nf/(time()-ptime)))
        ptime = time()
        nf = 0
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    elif key == 32:
        cv2.imwrite('testimage.png',frame);
    rval, frame = vc.read()

cv2.destroyWindow("preview")
cv2.destroyWindow("preview2")
vc.release()
# if vout:
#     vout.release()

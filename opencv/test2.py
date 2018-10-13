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
vc2 = cv2.VideoCapture(int(sys.argv[2]))
vc.set(3,int(sys.argv[3]))
vc.set(4,int(sys.argv[4]))
vc2.set(3,int(sys.argv[3]))
vc2.set(4,int(sys.argv[4]))
print(vc.get(3))
print(vc.get(4))

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

if vc2.isOpened(): # try to get the first frame
    rval2, frame2 = vc2.read()
else:
    rval2 = False

ptime = time()
nf = 0
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
glass_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
reye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
leye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')

face = None
eye = None
flost = 0
elost = 0
face2 = None
eye2 = None
flost2 = 0
elost2 = 0
while rval and rval2:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
    flost = flost+1
    elost = elost+1
    flost2 = flost+1
    elost2 = elost+1
    for f in faces:
        if face is not None:
            # print("Face: " + str(distance(f,face)))
            if not (1 < distance(f,face) < 20):
                continue
        face = f
        flost = 0
    for f in faces2:
        if face2 is not None:
            # print("Face: " + str(distance(f,face)))
            if not (1 < distance(f,face2) < 20):
                continue
        face2 = f
        flost2 = 0

    if flost < 5 and face is not None:
        (x,y,w,h) = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for e in eyes:
            if eye is not None:
                # print("Eyes: " + str(distance(eye,e)))
                if not (0 < distance(e,eye) < 10):
                    continue
            eye = e
            elost = 0

            if elost < 5 and eye is not None:
                (ex,ey,ew,eh) = eye
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            else:
                eye = None
    else:
        face = None
        eye = None
    if flost2 < 5 and face2 is not None:
        (x2,y2,w2,h2) = face2
        cv2.rectangle(frame2,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
        roi_gray2 = gray[y2:y2+h2, x2:x2+w2]
        roi_color2 = frame2[y2:y2+h2, x2:x2+w2]
        eyes = eye_cascade.detectMultiScale(roi_gray2)
        for e in eyes:
            if eye2 is not None:
                # print("Eyes: " + str(distance(eye,e)))
                if not (0 < distance(e,eye2) < 10):
                    continue
            eye2 = e
            elost2 = 0

            if elost2 < 5 and eye2 is not None:
                (ex2,ey2,ew2,eh2) = eye2
                cv2.rectangle(roi_color2,(ex2,ey2),(ex2+ew2,ey2+eh2),(0,255,0),2)
            else:
                eye2 = None
    else:
        face2 = None
        eye2 = None


    cv2.imshow("preview", frame)
    cv2.imshow("preview2", frame2)
    nf = nf + 1
    if time() - ptime > 5:
        print(str(nf/(time()-ptime)))
        ptime = time()
        nf = 0
    rval, frame = vc.read()
    rval2, frame2 = vc2.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    elif key == 32:
        cv2.imwrite('testimage.png',frame);

cv2.destroyWindow("preview")
vc.release()

import cv2
import numpy as np
import sys
from time import time


def rectcenter(rect):
    (x,y,w,h) = rect
    return (x+w/2,y+h/2)

def distance(c1, c2):
    return np.hypot(c1[0]-c2[0],c1[1]-c2[1])



cv2.namedWindow("preview")
cv2.namedWindow("previewblob")
vc = cv2.VideoCapture(int(sys.argv[1]))
vc.set(3,int(sys.argv[2]))
vc.set(4,int(sys.argv[3]))
rw = vc.get(3)
rh = vc.get(4)
print(vc.get(3))
print(vc.get(4))

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 5;
params.maxThreshold = 50;

# Filter by Area.
params.filterByArea = True
params.minArea = 1
params.maxArea = 3

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.7

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.88

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
detector = cv2.SimpleBlobDetector_create(params)
reye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

kernel = np.ones((5,5),np.uint8)

ptime = time()
nf = 0
thresh1 = 70
thresh2 = 135
while rval:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray3 = 255-gray

    gray2 = thresh1-gray
    # gray2 = cv2.equalizeHist(gray2)
    gray2 = cv2.erode(gray2,kernel)
    gray2 = cv2.GaussianBlur(gray2, (5,5), 6, 6)
    # gray2 = cv2.dilate(gray2,kernel)
    # blah,gray2 = cv2.threshold(gray2,thresh2,255,cv2.THRESH_BINARY)
    # gray2 = cv2.GaussianBlur(gray2, (5,5), 3, 3)

    eyes = reye_cascade.detectMultiScale(gray)
    # for e in eyes:
    #     (ex,ey,ew,eh) = e
    #     cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #     roi_gray = gray2[ey:ey+eh, ex:ex+ew]
    #     roi_color = frame[ey:ey+eh, ex:ex+ew]
    roi_gray = gray2
    rows = roi_gray.shape[0]
    circles = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT, 1, rows / 4,
                           param1=200, param2=10,
                           minRadius=10, maxRadius=20)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # print(distance(center,rectcenter(e)))
            if distance(center,rectcenter((0,rh/3,rw,rh))) > 200:
                continue
            # circle center
            cv2.circle(roi_gray, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(roi_gray, center, radius, (100, 0, 100), 3)

    blobframe = None
    keypoints = detector.detect(gray3)

    blobframe = cv2.drawKeypoints(gray3, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("preview", gray2)
    if blobframe is not None:
        cv2.imshow("previewblob",blobframe)

    nf = nf + 1
    if time() - ptime > 5:
        print(str(nf/(time()-ptime)))
        ptime = time()
        nf = 0
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key != -1:
    #     print(key)
        print("1: " + str(thresh1))
        print("2: " + str(thresh2))
    if key == 27: # exit on ESC
        break
    elif key == 32:
        cv2.imwrite('testimage.png',frame);
    elif key == 104:
        thresh1 = thresh1 + 5
    elif key == 106:
        thresh1 = thresh1 - 5
    elif key == 107:
        thresh2 = thresh2 + 5
    elif key == 108:
        thresh2 = thresh2 - 5

cv2.destroyWindow("preview")
vc.release()

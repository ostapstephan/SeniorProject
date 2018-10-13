from __future__ import print_function # WalabotAPI works on both Python 2 an 3.
import cv2
import numpy as np
import sys
from time import time
from sys import platform
from os import system
from imp import load_source
from os.path import join

if platform == 'win32':
        modulePath = join('C:/', 'Program Files', 'Walabot', 'WalabotSDK', 'python', 'WalabotAPI.py')
elif platform.startswith('linux'):
    modulePath = join('/usr', 'share', 'walabot', 'python', 'WalabotAPI.py')

wlbt = load_source('WalabotAPI', modulePath)
wlbt.Init()

def PrintSensorTargets(targets):
    system('cls' if platform == 'win32' else 'clear')
    if targets:
        for i, target in enumerate(targets):
            print('Target #{}:\nx: {}\ny: {}\nz: {}\namplitude: {}\n'.format(
                i + 1, target.xPosCm, target.yPosCm, target.zPosCm,
                target.amplitude))
    else:
        print('No Target Detected')

def SensorApp():
    # wlbt.SetArenaR - input parameters
    minInCm, maxInCm, resInCm = 30, 200, 3
    # wlbt.SetArenaTheta - input parameters
    minIndegrees, maxIndegrees, resIndegrees = -15, 15, 5
    # wlbt.SetArenaPhi - input parameters
    minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees = -60, 60, 5
    # Set MTI mode
    mtiMode = False
    # Configure Walabot database install location (for windows)
    wlbt.SetSettingsFolder()
    # 1) Connect : Establish communication with walabot.
    wlbt.ConnectAny()
    # 2) Configure: Set scan profile and arena
    # Set Profile - to Sensor.
    wlbt.SetProfile(wlbt.PROF_SENSOR)
    # Setup arena - specify it by Cartesian coordinates.
    wlbt.SetArenaR(minInCm, maxInCm, resInCm)
    # Sets polar range and resolution of arena (parameters in degrees).
    wlbt.SetArenaTheta(minIndegrees, maxIndegrees, resIndegrees)
    # Sets azimuth range and resolution of arena.(parameters in degrees).
    wlbt.SetArenaPhi(minPhiInDegrees, maxPhiInDegrees, resPhiInDegrees)
    # Moving Target Identification: standard dynamic-imaging filter
    filterType = wlbt.FILTER_TYPE_MTI if mtiMode else wlbt.FILTER_TYPE_NONE
    wlbt.SetDynamicImageFilter(filterType)
    # 3) Start: Start the system in preparation for scanning.
    wlbt.Start()
    if not mtiMode: # if MTI mode is not set - start calibrartion
        # calibrates scanning to ignore or reduce the signals
        wlbt.StartCalibration()
        while wlbt.GetStatus()[0] == wlbt.STATUS_CALIBRATING:
            wlbt.Trigger()

SensorApp()

def rectcenter(rect):
    (x,y,w,h) = rect
    return (x+w/2,y+h/2)

def distance(c1, c2):
    return np.hypot(c1[0]-c2[0],c1[1]-c2[1])

cv2.namedWindow("preview")
cv2.namedWindow("preview2")
cv2.namedWindow("previewblob")
vc = cv2.VideoCapture(int(sys.argv[1]))
vc2 = cv2.VideoCapture(int(sys.argv[2]))
vc.set(3,int(sys.argv[3]))
vc.set(4,int(sys.argv[4]))
rw = vc.get(3)
rh = vc.get(4)
# print(vc.get(3))
# print(vc.get(4))

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

if vc2.isOpened(): # try to get the first frame
    rval2, frame2 = vc2.read()
else:
    rval2 = False

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
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face = None
flost = 0

kernel = np.ones((5,5),np.uint8)

ptime = time()
nf = 0
thresh1 = 70
thresh2 = 135
record = False
numx = 0
numy = 0
while rval and rval2:
    flost = flost + 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray4, 1.3, 5)
    for f in faces:
        if face is not None:
            # print("Face: " + str(distance(f,face)))
            if not (1 < distance(f,face) < 20):
                continue
        face = f
        flost = 0
    if flost < 5 and face is not None:
        (x,y,w,h) = face
        cv2.rectangle(frame2,(x,y),(x+w,y+h),(255,0,0),2)
    else:
        face = None

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
    center = None
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
    cv2.imshow("preview2", frame2)
    if blobframe is not None:
        cv2.imshow("previewblob",blobframe)

    if record and face is not None and center is not None and blobframe is not None:
        (x,y,w,h) = face
        (px,py) = center
        if len(keypoints) > 4:
            keypoints = keypoints[0:4]
        else:
            keypoints = keypoints+(4-len(keypoints))*[cv2.KeyPoint(2,-1,-1)]

        appStatus, calibrationProcess = wlbt.GetStatus()
        # 5) Trigger: Scan(sense) according to profile and record signals
        # to be available for processing and retrieval.
        wlbt.Trigger()
        # 6) Get action: retrieve the last completed triggered recording
        targets = wlbt.GetSensorTargets()
        rasterImage, _, _, sliceDepth, power = wlbt.GetRawImageSlice()
        z = targets[0].zPosCm
        print(','.join([str(numx*274.28),str(numy*270),str(x),str(y),str(z),str(px),str(py)]+[str(x.pt[0]) for x in keypoints]+[str(x.pt[1]) for x in keypoints]))

    nf = nf + 1
    if time() - ptime > 5:
        # print(str(nf/(time()-ptime)))
        ptime = time()
        nf = 0
    rval, frame = vc.read()
    rval2, frame2 = vc2.read()
    key = cv2.waitKey(20)
    if key != -1:
        pass
    #     print(key)
        # print("1: " + str(thresh1))
        # print("2: " + str(thresh2))
    if key == 27: # exit on ESC
        wlbt.Stop()
        wlbt.Disconnect()
        break
    elif key == 32:
        # cv2.imwrite('testimage.png',frame);
        record = not record
        if not record:
            if numx == 7:
                numx = -1
                numy = numy + 1
            numx = numx + 1
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

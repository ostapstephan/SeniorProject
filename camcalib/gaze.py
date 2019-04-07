#!/usr/bin/env python
import cv2
import os
import subprocess as sp
import sys
import numpy as np
import time
# import datetime
from matrix import get_pupil_transformation_matrix
from threading import Thread
sys.path.append(os.path.abspath('../../TEST'))
sys.path.append(os.path.abspath('../../TEST/shared_modules'))
from pupil_detectors import Detector_3D
from methods import Roi
sys.path.append(os.path.abspath('../'))
# from calibrateHaar import calibrate
# from pbcvt import findPupilEllipse
# from params import pupil_tracker_params

from cameras import cam0mat as cameraMatrix0
from cameras import cam0dcoef as distCoeffs0

from cameras import cam1mat as cameraMatrix1
from cameras import cam1dcoef as distCoeffs1

cameraMatrix0 = np.array(cameraMatrix0)
distCoeffs0 = np.array(distCoeffs0)

cameraMatrix1 = np.array(cameraMatrix1)
distCoeffs1 = np.array(distCoeffs1)

# from cameras import cam1mat as cameraMatrix1
# from cameras import cam1dcoef as distCoeffs1

TIMEOUT = 10000
FFMPEG_BIN = "ffmpeg"
'''
This code will be able to open fast and low latency streams
and capture and save photos from webcams and network raspberry pi's
The Readme.txt in this dir will help with debugging
'''

class WebcamVideoStream:
    def __init__(self, src=None, fifo=None):
        # initialize the video camera stream and read the first frame
        # from the stream
        # self.stream = cv2.VideoCapture(src)
        # (self.grabbed, self.frame) = self.stream.read()

        ###
        if fifo == 'fifo0':
            self.height = 640
            self.width = 480
        elif fifo == 'fifo1':
            self.height = 480
            self.width = 640
        else:
            print('error please specify what camera type ')
            raise (Exception)

        if not fifo:
            fifo = 'fifo0'
            print("no input using fifo0")

        print("about to init command")
        command = [
            FFMPEG_BIN,
            '-i',
            fifo,
            '-pix_fmt',
            'bgr24',  # opencv requires bgr24 pixel format.
            '-vcodec',
            'rawvideo',
            '-an',
            '-sn',
            '-f',
            'image2pipe',
            '-'
        ]  # '-framerate', '100',

        print("about to sp.popen")
        self.pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=1024)

        print("about read first frame")
        try:
            raw_image = self.pipe.stdout.read(self.height * self.width * 3)
            self.image = np.fromstring(
                raw_image, dtype='uint8'
            ).reshape((self.height, self.width, 3))
        except Exception:
            self.image = np.zeros((self.height, self.width, 3))
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        print("starting thread")
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        print("starting while true loop")
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.pipe.kill()
                return
            raw_image = self.pipe.stdout.read(self.height * self.width * 3)
            self.image = np.fromstring(
                raw_image, dtype='uint8'
            ).reshape((self.height, self.width, 3))
            self.pipe.stdout.flush()
            # otherwise, read the next frame from the stream
            # (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.image

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

def draw_ellipse(
        img,
        center,
        axes,
        angle,
        startAngle,
        endAngle,
        color,
        thickness=3,
        lineType=cv2.LINE_AA,
        shift=10
):
    center = (int(round(center[0] * 2**shift)), int(round(center[1] * 2**shift)))
    axes = (int(round(axes[0] * 2**shift)), int(round(axes[1] * 2**shift)))
    cv2.ellipse(
        img,
        center,
        axes,
        angle,
        startAngle,
        endAngle,
        color,
        thickness,
        lineType,
        shift,
    )

class Frame(object):
    def __init__(self, camType):
        if camType == 0:
            self.height = 640
            self.width = 480
        elif camType == 1:
            self.height = 480
            self.width = 640

        self.gray = np.zeros((self.height, self.width))
        self.img = np.zeros((self.height, self.width, 3))
        self.timestamp = time.time()

def solveperp(objectPoints, imagePoints, cameraMatrix, distCoeffs, method):
    if method == 1:
        return cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
    elif method == 2:
        return cv2.solvePnPRansac(
            objectPoints, imagePoints, cameraMatrix, distCoeffs
        )
    else:
        return cv2.solveP3P(objectPoints, imagePoints, cameraMatrix, distCoeffs)

def draw_gaze(img, start, end, H, K, dist):
    # unit is mm
    try:
        rvec, _ = cv2.Rodrigues(H[:3,:3])
        tvec = H[:3,3]
        points = np.float32([
            start,
            end,
        ]).reshape(-1, 3)
        axisPoints, _ = cv2.projectPoints(points, rvec, tvec, K, dist)
        img = cv2.arrowedLine(
            img, tuple(axisPoints[0].ravel()), tuple(axisPoints[1].ravel()),
            (0, 255, 0), 3
        )
    except OverflowError:
        pass
    return img

def draw_plane(img, corners, H, K, dist):
    # unit is mm
    try:
        rvec, _ = cv2.Rodrigues(H[:3,:3])
        tvec = H[:3,3]
        points = np.float32(corners).reshape(-1, 3)
        axisPoints, _ = cv2.projectPoints(points, rvec, tvec, K, dist)
        img = cv2.arrowedLine(
            img, tuple(axisPoints[0].ravel()), tuple(axisPoints[1].ravel()),
            (0, 0, 255), 3
        )
        img = cv2.arrowedLine(
            img, tuple(axisPoints[1].ravel()), tuple(axisPoints[2].ravel()),
            (255, 0, 0), 3
        )
        img = cv2.arrowedLine(
            img, tuple(axisPoints[2].ravel()), tuple(axisPoints[3].ravel()),
            (255, 0, 0), 3
        )
        img = cv2.arrowedLine(
            img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()),
            (255, 0, 0), 3
        )
    except OverflowError:
        pass
    return img

def lineIntersection(planePoint, planeNormal, linePoint, lineDirection):
    if np.dot(planeNormal,lineDirection) == 0:
        return planePoint

    t = (np.dot(planeNormal,planePoint) - np.dot(planeNormal,linePoint)) / np.dot(planeNormal,lineDirection)
    return linePoint + t*lineDirection;

# class Roi(object):
# """this is a simple 2D Region of Interest class
# it is applied on numpy arrays for convenient slicing
# like this:
# roi_array_slice = full_array[r.view]
# # do something with roi_array_slice
# this creates a view, no data copying done
# """

# def __init__(self, array_shape):
# self.array_shape = array_shape
# self.lX = 0
# self.lY = 0
# self.uX = array_shape[1]
# self.uY = array_shape[0]
# self.nX = 0
# self.nY = 0

# open a named pipe for each pi and start listening
pipeinit0 = sp.Popen(['./r0.sh'], stdout=sp.PIPE)
pipeinit1 = sp.Popen(['./r1.sh'], stdout=sp.PIPE)

# start streaming from the pi to this computer
sshPi0 = sp.Popen(['ssh', 'pi@10.0.0.3', '-p', '6622', '~/stream.sh'], stdout=sp.PIPE)
vs0 = WebcamVideoStream(fifo="fifo0").start()
print()
print()
print('Fifo 0 started')
print()
print()
sshPi1 = sp.Popen(['ssh', 'pi@10.0.0.5', '~/stream.sh'], stdout=sp.PIPE)
vs1 = WebcamVideoStream(fifo="fifo1").start()
print()
print()
print('Fifo 1 started')
print()
print()
# i = 0
# j = 0

frame = Frame(0)
roi = Roi(frame.img.shape)
cv2.namedWindow('Video0')
cv2.namedWindow('Video1')
cv2.namedWindow('aruco')

vout0 = None
vout1 = None
if len(sys.argv) > 1:
    fourcc = cv2.VideoWriter_fourcc(*'x264')
    vout0 = cv2.VideoWriter('demo0.mp4', fourcc, 24.0, (frame.img.shape[1], frame.img.shape[0]))
    vout1 = cv2.VideoWriter('demo1.mp4', fourcc, 24.0, (frame.img.shape[0], frame.img.shape[1]))


## ACTUAL STUFF BELOW

pupil_detector = Detector_3D()
pupil_detector.set_2d_detector_property('pupil_size_max', 150)
# pupil_detector.set_2d_detector_property('pupil_size_min', 10)
# pupil_detector.set_2d_detector_property('ellipse_roundness_ratio', 0.1)
# pupil_detector.set_2d_detector_property('coarse_filter_max', 240)
# pupil_detector.set_2d_detector_property('intensity_range', 30)
# pupil_detector.set_2d_detector_property('canny_treshold', 200)
# pupil_detector.set_2d_detector_property('canny_ration', 3)
# pupil_detector.set_2d_detector_property('support_pixel_ratio_exponent', 3.0)
# pupil_detector.set_2d_detector_property('initial_ellipse_fit_treshhold', 1.5)
'''
'coarse_detection': True,
'coarse_filter_min': 128,
'coarse_filter_max': 280,
'intensity_range': 23,
'blur_size': 5,
'canny_treshold': 160,
'canny_ration': 2,
'canny_aperture': 5,
'pupil_size_max': 100,
'pupil_size_min': 10,
'strong_perimeter_ratio_range_min': 0.8,
'strong_perimeter_ratio_range_max': 1.1,
'strong_area_ratio_range_min': 0.6,
'strong_area_ratio_range_max': 1.1,
'contour_size_min': 5,
'ellipse_roundness_ratio': 0.1,
'initial_ellipse_fit_treshhold': 1.8,
'final_perimeter_ratio_range_min': 0.6,
'final_perimeter_ratio_range_max': 1.2,
'ellipse_true_support_min_dist': 2.5,
'support_pixel_ratio_exponent': 2.0
'''

objPoints = np.array(
    [(0, 0, 0), (536.575, 0, 0), (536.575, -361.95, 0), (0, -361.95, 0)]
)

UNITS_E = 1 # mm per box
UNITS_W = 14 # mm per box

Hoff = np.eye(4)
Hoff[:3,3] = np.array([-0.64, -1.28, 0.0])
HoffW = np.eye(4)
HoffW[:3,3] = np.array([0.0,0.0,0.0])
HEW = np.eye(4)
# R = np.array([78.69,90.0,180+39.67])
R = np.array([-14.0,40.0,143]) # ********** DONT DELETE
HEW[:3,:3] = cv2.Rodrigues(R)[0]
HEW[:3,3] = np.array([-58.58,-18.19,32.47])
# H90 = np.eye(4)
# H90[:3,:3] = cv2.Rodrigues(np.array([0.0,0.0,0.0]))[0]
# Z = 1000

markdict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters_create()
# TODO edit default params

rvecM = [0.0,0.0,0.0]
tvecM = [0.0,0.0,0.0]
plane = None

# aruco

def getNewArucoImg(): 
    markerSize = 93
    outimg = cv2.aruco.drawMarker(markdict, 5, markerSize) 
    height = 1050
    width = 1680
    bigPic  = np.ones((height,width)) 
    
    #random offset 
    yo = np.random.randint(0,width-markerSize)
    xo = np.random.randint(0,height-markerSize)   
    bigPic[xo:xo+markerSize,yo:yo+markerSize] = outimg
    
    
    return bigPic, (xo+markerSize/2,yo+markerSize/2)

aflag = True


## MAIN LOOP

while True:
    image0 = vs0.read()
    image1 = vs1.read()

    if image0 is not None:
        # image0 = cv2.rotate(image0, cv2.ROTATE_90_CLOCKWISE)
        frame.gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
        frame.img = image0.copy()
        prevImage = image0.copy()
        frame.timestamp = time.time()
    else:
        frame.img = prevImage.copy()
        frame.gray = cv2.cvtColor(prevImage, cv2.COLOR_BGR2GRAY)
        frame.timestamp = time.time()

    if image1 is not None:
        image1 = cv2.rotate(image1, cv2.ROTATE_180)
        prevImage1 = image1.copy() 

    else:
        image1 = prevImage1


    corners, ids, rejected = cv2.aruco.detectMarkers(image1, markdict, cameraMatrix=cameraMatrix1, distCoeff=distCoeffs1)
    # print(corners)
    # print('ids:',ids)
    image1 = cv2.aruco.drawDetectedMarkers(image1, corners, ids, (255,0,255))
    rvecsC, tvecsC, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 50, cameraMatrix1, distCoeffs1)
    # rvecsCs, tvecsCs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 20, cameraMatrix1, distCoeffs1)
     
    # print(rvecsC)
    # print("individual t vecs: ",tvecsC)
    if ids is not None and len(corners) == len(ids) == 5:
        imgPoints = np.array([corners[x] for x in ids.T[0].argsort()])
        plane =  np.array([tvecsC[x][0][:] for x in ids.T[0].argsort()])
        gazepoint = plane[4]
        plane = plane[:4]
        print("Monitor: ",plane)
        print("3d gaze point: ",gazepoint)
        # retval, rvecM, tvecM = solveperp(objPoints, imgPoints, cameraMatrix1, distCoeffs1, 1)

    result = pupil_detector.detect(frame, roi, True)
    draw_ellipse(
        frame.img, result['ellipse']['center'],
        [x / 2 for x in result['ellipse']['axes']], result['ellipse']['angle'], 0, 360,
        (255, 255, 0), 2
    )
    sphere = np.array(result['sphere']['center'])
    pupil = np.array(result['circle_3d']['center'])
    print("sphere: ", sphere)
    print("pupil: ", pupil)

    draw_gaze(
        frame.img, sphere, pupil, Hoff,
        cameraMatrix0, distCoeffs0
    )
    HEW[:3,:3] = cv2.Rodrigues(R)[0]
    H_all = Hoff @ HEW @ HoffW
    # print(H_all)
    sphere2 = H_all[:3,:3] @ sphere + H_all[:3,3]
    pupil2 = H_all[:3,:3] @ pupil + H_all[:3,3]
    pupil2[0] *= -1
    sphere2[0] *= -1
    # pupil2 *= UNITS_E/UNITS_W
    # sphere2 *= UNITS_E/UNITS_W
    # print("sphere2: ", sphere2)
    # print("pupil2: ", pupil2)
    gaze = pupil2-sphere2
    if plane is None:
        plane = objPoints.copy()
        plane[:,0] -= 536.575/2
        plane[:,1] += 361.95/2
        plane /= UNITS_W
        plane[:,2] = 10000

    # print("Plane: ",plane)
    draw_plane(image1, plane[0:4], np.eye(4), cameraMatrix1, distCoeffs1)
    gazeEnd = lineIntersection(plane[0], np.cross(plane[1]-plane[0],plane[2]-plane[1]), pupil, gaze)
    # gazeEnd2 = gazeEnd + 2*gaze
    # print("Cross: ", np.cross(plane[1]-plane[0],plane[2]-plane[1]))
    # print("GazeE: ", gazeEnd)
    # print("GazeE2: ", gazeEnd2)
    # print("R: ", R)
    # print("HoffW: ", HoffW)
    draw_gaze(
        image1, pupil, gazeEnd, np.eye(4),
        cameraMatrix1, distCoeffs1
    )
    # draw_gaze(
        # image1, gazeEnd, gazeEnd2, np.eye(4),
        # cameraMatrix1, distCoeffs1
    # )
    
    image1 = cv2.aruco.drawAxis(image1, cameraMatrix1, distCoeffs1,
            cv2.Rodrigues(H_all[:3,:3])[0], plane[0], 100)


    if image0 is not None:
        cv2.imshow('Video0', frame.img)
        if vout0:
            vout0.write(frame.img)
    if image1 is not None:
        cv2.imshow('Video1', image1)
        if vout1:
            vout1.write(image1)
    if aflag == True:
        aimg,(xxx,yyy)= getNewArucoImg()
        cv2.imshow("aruco", aimg) 
        print('the x and y of the center aruco img',xxx ,' ',yyy)
        aflag = False
     
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('h'):
        Hoff[:3,3][0] += 0.02
    elif key & 0xFF == ord('j'):
        Hoff[:3,3][0] -= 0.02
    elif key & 0xFF == ord('k'):
        Hoff[:3,3][1] += 0.02
    elif key & 0xFF == ord('l'):
        Hoff[:3,3][1] -= 0.02
    elif key & 0xFF == ord('t'):
        R[0] += 1.0
    elif key & 0xFF == ord('y'):
        R[0] -= 1.0
    elif key & 0xFF == ord('u'):
        R[1] += 1.0
    elif key & 0xFF == ord('i'):
        R[1] -= 1.0
    elif key & 0xFF == ord('o'):
        R[2] += 1.0
    elif key & 0xFF == ord('p'):
        R[2] -= 1.0
    elif key & 0xFF == ord('x'):
        HoffW[:3,3][0] += 1.02
    elif key & 0xFF == ord('c'):
        HoffW[:3,3][0] -= 1.02
    elif key & 0xFF == ord('v'):
        HoffW[:3,3][1] += 1.02
    elif key & 0xFF == ord('b'):
        HoffW[:3,3][1] -= 1.02
    elif key & 0xFF == ord('n'):
        HoffW[:3,3][2] += 1.02
    elif key & 0xFF == ord('m'):
        HoffW[:3,3][2] -= 1.02
    elif key & 0xFF == ord('a'):
        aflag = True
    # elif key & 0xFF == ord('z'):
        # Z += 1
    # elif key & 0xFF == ord('x'):
        # Z -= 1
    elif key == 32:  # spacebar will save the following images
        # cv2.imwrite('photos/0-'+str(time)+'.png', image0)
        # cv2.imwrite('photos/1-'+str(time)+'.png', image1)
        # time += 1
        pass

if vout0:
    vout0.release()
if vout1:
    vout1.release()
cv2.destroyAllWindows()
time.sleep(0.5)
vs0.stop()
vs1.stop()

#!/usr/bin/env python
import cv2
import os
import subprocess as sp
import sys
import numpy as np
import time
# import datetime
from threading import Thread
sys.path.append(os.path.abspath('../pupil/pupil_src/shared_modules'))
from pupil_detectors import Detector_3D
from methods import Roi
sys.path.append(os.path.abspath('../'))
# from calibrateHaar import calibrate
from pbcvt import findPupilEllipse
from params import pupil_tracker_params

from cameras import cam0mat as cameraMatrix0
from cameras import cam0dcoef as distCoeffs0

cameraMatrix0 = np.array(cameraMatrix0)
distCoeffs0 = np.array(distCoeffs0)

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


def draw_axis(img, R, t, K, dist):
    # unit is mm
    try:
        rotV, _ = cv2.Rodrigues(R)
        points = np.float32([
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 20],
            [0, 0, 0],
        ]).reshape(-1, 3)
        axisPoints, _ = cv2.projectPoints(points, rotV, t, K, dist)
        img = cv2.line(
            img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()),
            (255, 0, 0), 3
        )
        img = cv2.line(
            img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()),
            (0, 255, 0), 3
        )
        img = cv2.line(
            img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()),
            (0, 0, 255), 3
        )
    except OverflowError:
        pass
    return img


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
i = 0
j = 0

frame = Frame(0)

cv2.namedWindow('Video0')
cv2.namedWindow('Video1')

pupil_detector = Detector_3D()
pupil_detector.set_2d_detector_property('pupil_size_max', 80)
pupil_detector.set_2d_detector_property('pupil_size_min', 20)
pupil_detector.set_2d_detector_property('ellipse_roundness_ratio', 0.1)
# pupil_detector.set_2d_detector_property('coarse_filter_max', 240)
# pupil_detector.set_2d_detector_property('intensity_range', 30)
# pupil_detector.set_2d_detector_property('canny_treshold', 200)
pupil_detector.set_2d_detector_property('canny_ration', 3)
# pupil_detector.set_2d_detector_property('support_pixel_ratio_exponent', 3.0)
pupil_detector.set_2d_detector_property('initial_ellipse_fit_treshhold', 1.5)
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

roi = Roi(frame.img.shape)

while True:
    image0 = vs0.read()
    image1 = vs1.read()

    if image0 is not None:
        # image0 = cv2.rotate(image0, cv2.ROTATE_90_CLOCKWISE)
        frame.gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
        frame.img = image0
        frame.timestamp = time.time()

    # result = pupil_detector.detect(frame, roi , "algorithm" )
    # print(result['ellipse'])
    # print(result['circle_3d'])
    out = findPupilEllipse(frame.img, TIMEOUT, *pupil_tracker_params)
    draw_ellipse(
        frame.img, (out[0], out[1]), (out[2], out[3]), out[4], 0, 360, (0, 0, 0), 2
    )
    print(
        "ROI0!",
        ';'.join([str(x) for x in [roi.lX, roi.lY, roi.uX, roi.uY, roi.nX, roi.nY]])
    )
    result = pupil_detector.detect(frame, roi, "Roi")
    print(
        "ROI1!",
        ';'.join([str(x) for x in [roi.lX, roi.lY, roi.uX, roi.uY, roi.nX, roi.nY]])
    )
    draw_ellipse(
        frame.img, result['ellipse']['center'], result['ellipse']['axes'],
        result['ellipse']['angle'], 0, 360, (255, 255, 0), 2
    )
    print(result['circle_3d'])
    print(result['sphere'])
    # gaze = np.subtract(result['circle_3d']['center'], result['sphere']['center'])
    # gaze = gaze / np.linalg.norm(gaze)
    center = (-10, 10, -100)
    gaze = np.subtract((out[0], out[1], 80), center)
    gaze = gaze / np.linalg.norm(gaze)
    print(gaze)
    x = np.arccos(np.dot(gaze, [1, 0, 0])) * 180 / np.pi
    y = np.arccos(np.dot(gaze, [0, 1, 0])) * 180 / np.pi
    z = np.arccos(np.dot(gaze, [0, 0, 1])) * 180 / np.pi
    R, blah = cv2.Rodrigues(np.array([x, y, z]))
    t = result['sphere']['center']
    t = center
    print(t)
    print(R)
    draw_axis(frame.img, R, t, cameraMatrix0, distCoeffs0)

    if image0 is not None:
        cv2.imshow('Video0', frame.img)
    if image1 is not None:
        cv2.imshow('Video1', image1)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key == 32:  # spacebar will save the following images
        # cv2.imwrite('photos/0-'+str(time)+'.png', image0)
        # cv2.imwrite('photos/1-'+str(time)+'.png', image1)
        # time += 1
        pass

vs0.stop()
vs1.stop()
time.sleep(1)
cv2.destroyAllWindows()

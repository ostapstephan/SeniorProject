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

TIMEOUT = 10000
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

        # FMPEG_BIN = "ffmpeg"

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
    center = (
        int(round(center[0] * 2**shift)), int(round(center[1] * 2**shift))
    )
    axes = (int(round(axes[0] * 2**shift)), int(round(axes[1] * 2**shift)))
    cv2.ellipse(
        img, center, axes, angle, startAngle, endAngle, color, thickness,
        lineType, shift
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

FFMPEG_BIN = "ffmpeg"
# open a named pipe for each pi and start listening
pipeinit0 = sp.Popen(['./r0.sh'], stdout=sp.PIPE)
pipeinit1 = sp.Popen(['./r1.sh'], stdout=sp.PIPE)

# start streaming from the pi to this computer
sshPi0 = sp.Popen(
    ['ssh', 'pi@10.0.0.3', '-p', '6622', '~/stream.sh'], stdout=sp.PIPE
)
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

roi = Roi(frame.img.shape)

while True:
    image0 = vs0.read()
    image1 = vs1.read()

    if image0 is not None:
        # image0 = cv2.rotate(image0, cv2.ROTATE_90_CLOCKWISE)
        frame.gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
        frame.img = image0
        frame.timestamp = time.time()

    try:
        # result = pupil_detector.detect(frame, roi , "algorithm" )
        # print(result['ellipse'])
        # print(result['circle_3d'])
        out = findPupilEllipse(frame.img, TIMEOUT, *pupil_tracker_params)
        draw_ellipse(
            frame.img, (out[0], out[1]), (out[2], out[3]), out[4], 0, 360,
            (0, 0, 0), 2
        )
        result = pupil_detector.detect(frame, roi, "roi")
        draw_ellipse(
            frame.img, result['ellipse']['center'], result['ellipse']['axes'],
            result['ellipse']['angle'], 0, 360, (255, 255, 0), 2
        )
        print(result['circle_3d'])
    except Exception:
        print('nah fam')

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

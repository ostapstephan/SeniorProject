import cv2
import subprocess as sp
import sys
import numpy

# import the necessary packages
import datetime
from threading import Thread
import cv2
 
class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
 
    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self
 
    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()
 
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
 
    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()
 
    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src=None,fifo=None):
        # initialize the video camera stream and read the first frame
        # from the stream
        # self.stream = cv2.VideoCapture(src)
        # (self.grabbed, self.frame) = self.stream.read()
                
        ###

        FMPEG_BIN = "ffmpeg"        
        
        if not fifo:
            fifo = 'fifo0'
            print("no input using fifo0") 
        
        
        print("about to init command") 
        command = [ FFMPEG_BIN,'-i', fifo, '-pix_fmt', 'bgr24',     # opencv requires bgr24 pixel format.
                '-vcodec', 'rawvideo','-an','-sn',
                '-f', 'image2pipe', '-']                            # '-framerate', '100',
        
        print("about to sp.popen") 
        self.pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=1024)

        print("about read first frame") 
        try:
            raw_image = self.pipe.stdout.read(640*480*3)
            self.image =  numpy.fromstring(raw_image, dtype='uint8').reshape((480,640,3))
        except:
            self.image = numpy.zeros((480,640,3))
        print(self.image.shape)
        ### 

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
            
            raw_image = self.pipe.stdout.read(640*480*3)
            self.image =  numpy.fromstring(raw_image, dtype='uint8').reshape((480,640,3))
            
            self.pipe.stdout.flush()
            
            # otherwise, read the next frame from the stream
            # (self.grabbed, self.frame) = self.stream.read()
 
    def read(self):
        # return the frame most recently read
        return self.image
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


'''
# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
    help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
    help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
 
# loop over some frames...this time using the threaded stream
while fps._numFrames < args["num_frames"]: #while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    # frame = imutils.resize(frame, width=400)
 
    # check to see if the frame should be displayed to our screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
    # update the FPS counter
    fps.update()
 
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
'''
# the above code was given by the threading post
#########################

FFMPEG_BIN = "ffmpeg"
#open a named pipe and start listening
pipeinit0 = sp.Popen(['./r0.sh'], stdout = sp.PIPE )
pipeinit1 = sp.Popen(['./r1.sh'], stdout = sp.PIPE )

#start streaming from the pi to this computer
sshPi0 = sp.Popen(['ssh', 'pi@10.0.0.3', '-p', '6622', '~/stream.sh'], stdout = sp.PIPE )

'''
# cam 1 
fifo = 'fifo0'
command = [ FFMPEG_BIN,'-i', fifo,'-pix_fmt', 'bgr24',      # opencv requires bgr24 pixel format.
        '-vcodec', 'rawvideo','-an','-sn',
        '-framerate', '100',
        '-f', 'image2pipe', '-']    
pipe0 = sp.Popen(command, stdout = sp.PIPE, bufsize=1024)

# cam 2
fifo = 'fifo1'
command = [ FFMPEG_BIN,
        '-i', fifo,            # fifo is the named pipe
        '-pix_fmt', 'bgr24',      # opencv requires bgr24 pixel format.
        '-vcodec', 'rawvideo',
        '-an','-sn',              # we want to disable audio processing (there is no audio)
        '-framerate', '100',
        '-f', 'image2pipe', '-']    
pipe1 = sp.Popen(command, stdout = sp.PIPE, bufsize=1024)
'''

vs0 = WebcamVideoStream(fifo = "fifo0").start() 
print()
print('hoooooo')
print()
print()

sshPi1 = sp.Popen(['ssh', 'pi@10.0.0.5', '~/stream.sh'], stdout = sp.PIPE )

vs1 = WebcamVideoStream(fifo = "fifo1").start()
print()
print()
print()
print('heyyyy')

i=0
cap = [None,None,None,None]
j=0

while i < 4: 
    try:
        cap[i] = cv2.VideoCapture(int(j))
        print("accepted:",i,j)
        i+=1
    except:
        print(i,j)
        pass
    j+=1
# Cam 3 and 4


# cap3.set(3,1280)
# cap3.set(4,720)

cv2.namedWindow('Video0')
cv2.namedWindow('Video1')
cv2.namedWindow('Video2')
cv2.namedWindow('Video3')
cv2.namedWindow('Video4')
cv2.namedWindow('Video5')

time = 0


while True:    
    ''' 
    # Capture frame-by-frame
    raw_image0 = pipe0.stdout.read(640*480*3)
    raw_image1 = pipe1.stdout.read(640*480*3)

    # transform the byte read into a numpy array
    image0 =  numpy.fromstring(raw_image0, dtype='uint8').reshape((480,640,3))
    image1 =  numpy.fromstring(raw_image1, dtype='uint8').reshape((480,640,3))
    # Notice how height is specified first and then width
    '''

    image0 = vs0.read()
    image1 = vs1.read()
    _, image2 = cap[0].read()
    _, image3 = cap[1].read() 
    _, image4 = cap[2].read()
    _, image5 = cap[3].read()


    if image0 is not None:
        cv2.imshow('Video0', image0)
    if image1 is not None:
        cv2.imshow('Video1', image1)
    if image2 is not None:
        cv2.imshow('Video2', image2)
    if image3 is not None:
        cv2.imshow('Video3', image3)
    if image4 is not None:
        cv2.imshow('Video4', image4)
    if image5 is not None:
        cv2.imshow('Video5', image5)
 
    key = cv2.waitKey(1) 
    if key & 0xFF == ord('q'):
        vs0.stop()
        break

    elif key == 32:  # spacebar
        cv2.imwrite('photos/cam0-'+str(time)+'.png', image0)
        cv2.imwrite('photos/cam1-'+str(time)+'.png', image1)
        cv2.imwrite('photos/cam2-'+str(time)+'.png', image2)
        cv2.imwrite('photos/cam3-'+str(time)+'.png', image3)
        cv2.imwrite('photos/cam4-'+str(time)+'.png', image4)
        cv2.imwrite('photos/cam5-'+str(time)+'.png', image5)
        time += 1

    # pipe0.stdout.flush()
    # pipe1.stdout.flush()

cap[0].release()
cap[1].release()
cap[2].release()
cap[3].release()

cv2.destroyAllWindows()

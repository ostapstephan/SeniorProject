import cv2
import subprocess as sp
import sys
import numpy
import datetime
from threading import Thread
'''
This code will be able to open fast and low latency streams 
and capture and save photos from webcams and network raspberry pi's
The Readme.txt in this dir will help with debugging

Look into the fifo height width cuz i messed with that and im not sure if its
broken
'''
class WebcamVideoStream:
    def __init__(self,  camType,  src=None,fifo=None):
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
            raise(Exception)
            
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
            raw_image = self.pipe.stdout.read(self.height*self.width*3)
            self.image =  numpy.fromstring(raw_image,dtype='uint8').reshape((self.height,self.width,3))
        except:
            self.image = numpy.zeros((self.height,self.width,3))
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
            raw_image = self.pipe.stdout.read(self.height*self.width*3)
            self.image =  numpy.fromstring(raw_image,dtype='uint8').reshape((self.height,self.width,3))
            self.pipe.stdout.flush()
            # otherwise, read the next frame from the stream
            # (self.grabbed, self.frame) = self.stream.read()
 
    def read(self):
        # return the frame most recently read
        return self.image
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

# the above code was given by the threading post
#########################

FFMPEG_BIN = "ffmpeg"
#open a named pipe for each pi and start listening
pipeinit0 = sp.Popen(['./r0.sh'], stdout = sp.PIPE )
pipeinit1 = sp.Popen(['./r1.sh'], stdout = sp.PIPE )

#start streaming from the pi to this computer
sshPi0 = sp.Popen(['ssh', 'pi@10.0.0.3', '-p', '6622', '~/stream.sh'], stdout = sp.PIPE )
vs0 = WebcamVideoStream(fifo = "fifo0").start() 
print()
print()
print('Fifo 0 started')
print()
print()
sshPi1 = sp.Popen(['ssh', 'pi@10.0.0.5', '~/stream.sh'], stdout = sp.PIPE )
vs1 = WebcamVideoStream(fifo = "fifo1").start()
print()
print()
print('Fifo 1 started')
print()
print()
i=0
cap = [None,None,None,None]
j=0
# i is the num cameras to open streams for
while i < 3: 
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
cv2.namedWindow('Video2')
cv2.namedWindow('Video3')
cv2.namedWindow('Video4')
# cv2.namedWindow('Video5')

time = 0


while True:     
    image0 = vs0.read()
    image1 = vs1.read()
    _, image2 = cap[0].read()
    _, image3 = cap[1].read() 
    _, image4 = cap[2].read()
    # _, image5 = cap[3].read()

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
    # if image5 is not None:
        # cv2.imshow('Video5', image5)
 
    key = cv2.waitKey(1) 
    if key & 0xFF == ord('q'): 
        break
    
    elif key == 32:  # spacebar will save the following images
        #cv2.imwrite('calb/0-'+str(time)+'.png', image0)
        cv2.imwrite('photos/0-'+str(time)+'.png', image0)
        cv2.imwrite('photos/1-'+str(time)+'.png', image1)
        cv2.imwrite('photos/2-'+str(time)+'.png', image2)
        cv2.imwrite('photos/3-'+str(time)+'.png', image3)
        cv2.imwrite('photos/4-'+str(time)+'.png', image4)
        # cv2.imwrite('photos/5-'+str(time)+'.png', image5)
        time += 1

cap[0].release()
cap[1].release()
cap[2].release()
#cap[3].release()
vs0.stop()
vs1.stop()

cv2.destroyAllWindows()





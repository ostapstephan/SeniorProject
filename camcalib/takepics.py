#!/usr/bin/env python
import cv2
import sys
import subprocess as sp
import numpy
# cap0,cap1,cap2,cap3= 0,0,0,0

''' Depreciated use streamAndCapture '''


FFMPEG_BIN = "ffmpeg"

# cap0 = cv2.VideoCapture('udp://10.0.0.2:5001')
# cv2.namedWindow('test0')

# cap1 = cv2.VideoCapture('udp://10.0.0.2:5002')
# cv2.namedWindow('test1')

command0 = [ FFMPEG_BIN,'-i', 'fifo1',   # fifo is the named pipe
        '-pix_fmt', 'bgr24',            # opencv requires bgr24 pixel format.
        '-vcodec', 'rawvideo',
        '-an','-sn',              # we want to disable audio processing (there is no audio)
        '-f', 'image2pipe', '-']    
pipe0 = sp.Popen(command0, stdout = sp.PIPE, bufsize=10**8)  

command1 = [ FFMPEG_BIN,'-i', 'fifo0',   # fifo is the named pipe
        '-pix_fmt', 'bgr24',            # opencv requires bgr24 pixel format.
        '-vcodec', 'rawvideo',
        '-an','-sn',              # we want to disable audio processing (there is no audio)
        '-f', 'image2pipe', '-']    
pipe1 = sp.Popen(command1, stdout = sp.PIPE, bufsize=10**8) #buf size TODO make sure its not too big or small 

cap2 = cv2.VideoCapture(int(sys.argv[1]))
cv2.namedWindow('test2')

cap3 = cv2.VideoCapture(int(sys.argv[2]))
cv2.namedWindow('test3')


def read0():
    # Capture frame-by-frame
    raw_image0 = pipe0.stdout.read(640*480*3)
    # transform the byte read into a numpy array
    image0 =  numpy.fromstring(raw_image0, dtype='uint8')
    image0 = image0.reshape((480,640,3))          # Notice how height is specified first and then width
    return image0

def read1():
    # Capture frame-by-frame
    raw_image1 = pipe1.stdout.read(640*480*3)
    # transform the byte read into a numpy array
    image1 =  numpy.fromstring(raw_image0, dtype='uint8')
    image1 = image1.reshape((480,640,3))          # Notice how height is specified first and then width
    return image1


cap3.set(3,1280)
cap3.set(4,480)

time = 0
while(True):
    # ret, frame0 = cap0.read()
    # ret, frame1 = cap1.read() 
    # frame0 = read0()
    # frame1 = read1()
    

    # this is the same as what we are doing in the read0/1 but outside the funtion
    
    raw_image1 = pipe1.stdout.read(640*480*3)
    # transform the byte read into a numpy array
    image1 =  numpy.fromstring(raw_image0, dtype='uint8')
    frame1 = image1.reshape((480,640,3))  
    
    raw_image0 = pipe0.stdout.read(640*480*3)
    # transform the byte read into a numpy array
    image0 =  numpy.fromstring(raw_image0, dtype='uint8')
    frame0 = image0.reshape((480,640,3))          # Notice how height is specifie


    ret, frame2 = cap2.read()
    ret, frame3 = cap3.read()

    # image = getImage()
    if image is not None:
        cv2.imshow('Video', image)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
    # pipe.stdout.flush()

    # frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
    try:
        cv2.imshow('test0', frame0)
    except:
        print("imshow failed frame 0")

    try:
        cv2.imshow('test1', frame1)
    except:
        print("imshow failed, frame1" )
    
    
    try:
        cv2.imshow('test2', frame2)
    except:
        print("imshow failed frame2" )

    try:
        cv2.imshow('test3', frame3)
    except:
        print("imshow failed frame3" )
        
    key = cv2.waitKey(1) 
    if key & 0xFF == ord('q'):
        break
    elif key == 32:  # spacebar
        cv2.imwrite('photos/0-'+str(time)+'.png', frame0)
        cv2.imwrite('photos/1-'+str(time)+'.png', frame1)
        cv2.imwrite('photos/2-'+str(time)+'.png', frame2)
        cv2.imwrite('photos/3-'+str(time)+'.png', frame3)
        time += 1

cap0.release()
cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()


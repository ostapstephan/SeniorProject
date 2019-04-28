#!/usr/bin/env python
import cv2
import os
import subprocess as sp
import sys
import numpy as np
import time
import pickle as pk
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


from sklearn.model_selection import train_test_split
import sklearn.linear_model
import sklearn.utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb


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

markdict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.adaptiveThreshConstant = 10

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

def drawArucoImg(xo,yo): 
    markerSize = 93
    outimg = cv2.aruco.drawMarker(markdict, 5, markerSize) 
    height = 1050
    width = 1680
    bigPic  = np.ones((height,width)) 
    xo = int(xo)
    yo = int(yo)
    bigPic[xo:xo+markerSize,yo:yo+markerSize] = outimg
    return bigPic 

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
        shift=10):
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

def useHomog(plane,pupil,sphere,R,t):
    H = np.eye(4)
    H[:3,:3] = cv2.Rodrigues(R)[0]
    H[:3,3]= t 
    
    sphere2 = H[:3,:3] @ sphere + H[:3,3] # THESE TWO LINES
    pupil2  = H[:3,:3] @ pupil  + H[:3,3]   # THATS THIS ONE TOO 
    gaze = pupil2-sphere2  
    gazeEnd = lineIntersection(plane[0],np.cross(plane[1]-plane[0],plane[2]-plane[1]), pupil2, gaze) 
    return gazeEnd  #predi

def lineIntersection(planePoint, planeNormal, linePoint, lineDirection):  #THIS FUNCTION
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
    vout0 = cv2.VideoWriter(sys.argv[1]+'0.mp4', fourcc, 24.0, (frame.img.shape[1], frame.img.shape[0]))
    vout1 = cv2.VideoWriter(sys.argv[2]+'1.mp4', fourcc, 24.0, (frame.img.shape[0], frame.img.shape[1]))


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

# Hoff = np.eye(4)
# Hoff[:3,3] = np.array([-0.64, -1.28, 0.0])
# HoffW = np.eye(4)
# HoffW[:3,3] = np.array([0.0,0.0,0.0])



Hoff = np.eye(4)
Hoff[:3, 3] = np.array([-1.06, -1.28, 0.0])
HoffW = np.eye(4)
HoffW[:3, 3] = np.array([-168.0, -100.0, -235.0])



HEW = np.eye(4)
# R = np.array([78.69,90.0,180+39.67])
R = np.array([-14.0,40.0,143]) # ********** DONT DELETE
HEW[:3,:3] = cv2.Rodrigues(R)[0]
HEW[:3,3] = np.array([-58.58,-18.19,32.47])
# H90 = np.eye(4)
# H90[:3,:3] = cv2.Rodrigues(np.array([0.0,0.0,0.0]))[0]
# Z = 1000

# HEATMAP
def gkern(kernlen, sigma):
    # First a 1-D  Gaussian
    lim = kernlen // 2 + (kernlen % 2) / 2
    t = np.linspace(-lim, lim, kernlen)
    bump = np.exp(-0.25 * (t / sigma)**2)
    bump /= np.trapz(bump)  # normalize the integral to 1

    # make a 2-D kernel out of it
    return bump[:, np.newaxis] * bump[np.newaxis, :]


def convertFeat(f): 
    # print(f)
    feat = np.zeros((1, 26 ))
    feat[0][:12]   =          f['corners'].flatten()
    feat[0][12:15] = np.array(f['eyeCenter']['center']) 
    feat[0][15]    =          f['eyeCenter']['radius'] 
    feat[0][16:19] = np.array(f['pupilCenter']['center']) 
    feat[0][19:22] = np.array(f['pupilCenter']['normal']) 
    feat[0][22]    =          f['pupilCenter']['radius']

    plane = f['corners']
    pupil = f['pupilCenter']['center']
    sphere= f['eyeCenter']['center']  
    t = np.array([-168.0, -100.0, -235.0])
    R = np.array([-14.0,40.0,143])     



    feat[0][23:] = useHomog(plane,pupil,sphere,R,t)
    return feat

radius = 200
sigma = 30
gain = 500
decay = 1.007
mask = gkern(2 * radius + 1, sigma) * gain


img0 = np.zeros((1050, 1680, 3))
img1 = np.zeros((1050, 1680, 3))



cv2.namedWindow('heatmap')
curpos = [int(img1.shape[0] / 2), int(img1.shape[1] / 2)]


# aruco
rvecM = [0.0,0.0,0.0]
tvecM = [0.0,0.0,0.0]
plane = None
aflag = False

########################################
# open and unpickle
with open ("training/databig.pickle", 'rb') as handle: 
    Data = pk.load(handle) 

feat = np.zeros((len(Data), 26 ))
lab  = np.zeros((len(Data), 2 ))
ones = np.ones((len(Data),1))

for i in range(len(Data)): 
    feat[i][:12]   =          Data[i][0]['corners'].flatten()
    feat[i][12:15] = np.array(Data[i][0]['eyeCenter']['center']) 
    feat[i][15]    =          Data[i][0]['eyeCenter']['radius'] 
    feat[i][16:19] = np.array(Data[i][0]['pupilCenter']['center']) 
    feat[i][19:22] = np.array(Data[i][0]['pupilCenter']['normal']) 
    feat[i][22]    =          Data[i][0]['pupilCenter']['radius']

    plane = Data[i][0]['corners']
    pupil = Data[i][0]['pupilCenter']['center']
    sphere= Data[i][0]['eyeCenter']['center']  
    t = np.array([-0.64, -1.28, 0.0])
    R = np.array([-14.0,40.0,143])     
    feat[i][23:] = useHomog(plane,pupil,sphere,R,t)

    lab[i][:] =np.array( Data[i][1]['2dpoint'] )

    if i ==1:
        print(feat[1],lab[1])

def mae(a,b): # mean absolute error
    return np.mean(abs(a.flatten()-b.flatten()))
def mse(a,b): # mean-squared error, input: Nx2
    #return np.sqrt(np.mean(abs(a.flatten()-b.flatten())**2))
    #return np.linalg.norm(a-b)
    return np.sqrt(np.mean(np.linalg.norm(a-b,axis=1)**2))

# shuffle & split
feat,lab = sklearn.utils.shuffle(feat,lab)
X_train, X_test, y_train, y_test = train_test_split(feat, lab, test_size=0.1, random_state=0)

# XGBoost
# '''
dtrainx = xgb.DMatrix(X_train,y_train[:,0])
dtest = xgb.DMatrix(X_test)
paramsx = {'eta': 0.1, 'gamma': 1.0,
               'min_child_weight': 0.1, 'max_depth': 6}
xgb_modelx = xgb.train(paramsx, dtrainx, num_boost_round=100)
dtrainy = xgb.DMatrix(X_train,y_train[:,1])
paramsy = {'eta': 0.1, 'gamma': 1.0,
        'min_child_weight': 0.1, 'max_depth': 6}
xgb_modely = xgb.train(paramsy, dtrainy, num_boost_round=100)


predx = xgb_modelx.predict(dtest)
predy = xgb_modely.predict(dtest)
'''
# polynomial regression
poly = PolynomialFeatures(degree=2,include_bias=False)
X_poly = poly.fit_transform(X_train)
poly_reg = sklearn.linear_model.LinearRegression()
poly_reg.fit(X_poly,y_train)

y_poly_est = poly_reg.predict(poly.transform(X_test))
#print(np.hstack((y_poly_est,y_test)))
print('poly MAE:',[ mae(y_poly_est[:,0],y_test[:,0]), mae(y_poly_est[:,1],y_test[:,1])])
print('poly MSE: ',mse(y_poly_est,y_test))
'''

########################################


xoff = 1680/2
yoff = 1050/2
factorx = 4
factory = 4
## MAIN LOOP
outData = []
count = 0
gazepoint= None

while True:
    features = {}
    labels = {}
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
        # print("Monitor: ",plane)
        features['corners'] = plane
        # print("3d gaze point: ",gazepoint)
        labels['3dpoint'] = gazepoint

    result = pupil_detector.detect(frame, roi, True)
    draw_ellipse(
        frame.img, result['ellipse']['center'],
        [x / 2 for x in result['ellipse']['axes']], result['ellipse']['angle'], 0, 360,
        (255, 255, 0), 2
    )
    sphere = np.array(result['sphere']['center'])
    pupil = np.array(result['circle_3d']['center'])
    # print("sphere: ", sphere)
    # print("pupil: ", pupil)
    features['eyeCenter'] = result['sphere']
    features['pupilCenter'] =  result['circle_3d']
   

    if 'corners' in features and 'eyeCenter' in features and 'pupilCenter'in features:
        feat = convertFeat(features) 
        
        dtest = xgb.DMatrix(feat)
        predx = xgb_modelx.predict(dtest)
        predy = xgb_modely.predict(dtest)
        curpos =  int(predx),int(predy) 
       
        # y_poly_est = poly_reg.predict(poly.transform(feat))
        # print(y_poly_est)
        # curpos =[int(y_poly_est[0][0]), int(y_poly_est[0][1])]
        # print('curpos',curpos)
    else:
        print('corners' in features)
        curpos = None
    
    
    if curpos is not None:
        #heatmap
        xn = max(curpos[0] - radius, 0)
        yn = max(curpos[1] - radius, 0)
        xm = min(curpos[0] + radius + 1, img1.shape[0])
        ym = min(curpos[1] + radius + 1, img1.shape[1])
        kxn = radius - (curpos[0] - xn)
        kyn = radius - (curpos[1] - yn)
        kxm = radius + xm - curpos[0]
        kym = radius + ym - curpos[1]
        # print(curpos)
        # print((xn, yn), ' ', (xm, ym))
        # print((kxn, kyn), ' ', (kxm, kym))
        img1[xn:xm, yn:ym, 0] += mask[kxn:kxm, kyn:kym]
        img1[xn:xm, yn:ym, 1] -= mask[kxn:kxm, kyn:kym] / 4
        img1[xn:xm, yn:ym, 2] -= mask[kxn:kxm, kyn:kym] / 2

    img1[:, :, :] /= decay
    cv2.imshow('heatmap', img0 + img1)


    draw_gaze(
        frame.img, sphere, pupil, Hoff,
        cameraMatrix0, distCoeffs0
    )


    HEW[:3,:3] = cv2.Rodrigues(R)[0]
    H_all = Hoff @ HEW @ HoffW
    # print(H_all)
    sphere2 = H_all[:3,:3] @ sphere + H_all[:3,3] # THESE TWO LINES
    pupil2 = H_all[:3,:3] @ pupil + H_all[:3,3]   # THATS THIS ONE TOO
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
    gazeEnd = lineIntersection(plane[0],np.cross(plane[1]-plane[0],plane[2]-plane[1]), pupil2, gaze) #TODO fix the thing to be either pupil 0 k

    draw_gaze(
        image1, pupil, gazeEnd, np.eye(4),
        cameraMatrix1, distCoeffs1
    )
    
    image1 = cv2.aruco.drawAxis(image1, cameraMatrix1, distCoeffs1,
            cv2.Rodrigues(H_all[:3,:3])[0], plane[0], 100)

    # gazepoint2d = np.abs(plane[1] - gazeEnd)[:2] * 2
    # features["2dpoint"] = gazepoint2d

    if image0 is not None:
        cv2.imshow('Video0', frame.img)
        if vout0:
            vout0.write(frame.img)
    if image1 is not None:
        cv2.imshow('Video1', image1)
        if vout1:
            vout1.write(image1)
   
    if aflag == True:
        if xoff >= 1680-96 or xoff<=1:
            factorx*=-1
        if yoff >= 1050-96 or yoff<=1:
            factory*=-1
        xoff += factorx
        yoff += factory
        # print(xoff,yoff) 
        aimg = drawArucoImg(yoff,xoff)
        cv2.imshow("aruco", aimg) 
        
        # cv2.imwrite('training/img0-'+str(count)+'.png', frame.img)
        # cv2.imwrite('training/img1-'+str(count)+'.png', image1) 
        # count += 1
        labels["2dpoint"]=(xoff+(93/2),yoff +(93/2))

        if 'corners' in features and 'eyeCenter' in features and 'pupilCenter'in features and '3dpoint' in labels and '2dpoint' in labels:
            outData.append((features,labels))
            print("appended")
            print((features,labels))
        else:
            print("Didn't quite catch that")

    # if aflag == True:
        # aimg,(xxx,yyy)= getNewArucoImg()
        # cv2.imshow("aruco", aimg) 
        # # print('the x and y of the center aruco img',xxx ,' ',yyy)
        # aflag = False
    
    
    
    
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
        aflag = not aflag 
    # elif key & 0xFF == ord('z'):
        # Z += 1
    # elif key & 0xFF == ord('x'):
        # Z -= 1
    # elif key == 32:  # spacebar will save the following images
       # pass 

with open('./training/data'+str(time.time())+'.pickle', 'wb') as handle:
    pk.dump(outData, handle, protocol=pk.HIGHEST_PROTOCOL)
    print("Saved")     

if vout0:
    vout0.release()
if vout1:
    vout1.release()
cv2.destroyAllWindows()
time.sleep(0.5)
vs0.stop()
vs1.stop()

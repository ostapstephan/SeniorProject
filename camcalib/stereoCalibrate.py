import numpy as np
import cv2
import glob
import sys

from cameras import cam2mat as cameraMatrix0
from cameras import cam2dcoef as distCoeffs0

from cameras import cam3mat as cameraMatrix1
from cameras import cam3dcoef as distCoeffs1

cameraMatrix0 = np.array(cameraMatrix0)
distCoeffs0 = np.array(distCoeffs0)
cameraMatrix1 = np.array(cameraMatrix1)
distCoeffs1 = np.array(distCoeffs1)
# Run an intrinsic checkerboard calibration for the camera of your specification 
# to run this just run 
# python camCalibrate.py (number of the camera you want to calibrate in the data folder)


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cbrow, cbcol = 7, 9
# this is for 7,11 because you have to have inlier points.
# https://stackoverflow.com/questions/31249037/calibrating-webcam-using-python-and-opencv-error/36441746

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow * cbcol, 3), np.float32)
print(objp.shape)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)
print(objp.shape)

# Arrays to store object points and image points from all the images.
objpoints0 = []  # 3d point in real world space
imgpoints0 = []  # 2d points in image plane.
objpoints1 = []  # 3d point in real world space
imgpoints1 = []  # 2d points in image plane.

# images = glob.glob('photosGoodish/' + sys.argv[1] + '*.png')

# print(images)
# keep track of how many were detected out of the total images looked at
i, j = 0, 0

for index in range(100):

    i += 1
    fname0 = 'photosGoodish/' + sys.argv[1] + '-' + str(index) + '.png'
    fname1 = 'photosGoodish/' + sys.argv[2] + '-' + str(index) + '.png'
    img0 = cv2.imread(fname0)
    img1 = cv2.imread(fname0)
    if img0 is None or img1 is None:
        continue

    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret0, corners0 = cv2.findChessboardCorners(gray0, (cbcol, cbrow), None)
    ret1, corners1 = cv2.findChessboardCorners(gray1, (cbcol, cbrow), None)
    # If found, add object points, image points (after refining them)
    if ret0 and ret1:
        objpoints0.append(objp)
        objpoints1.append(objp)
        j += 1

        # image, corners, winSize, zeroZone, criteria
        corners0 = cv2.cornerSubPix(gray0,
                                    corners0,
                                    (11, 11),
                                    (-1, -1),
                                    criteria)
        imgpoints0.append(corners0)
        corners1 = cv2.cornerSubPix(gray1,
                                    corners1,
                                    (11, 11),
                                    (-1, -1),
                                    criteria)
        imgpoints1.append(corners1)

        # Draw and display the corners
        img0 = cv2.drawChessboardCorners(img0, (cbcol, cbrow), corners0, ret0)
        img1 = cv2.drawChessboardCorners(img1, (cbcol, cbrow), corners1, ret1)
        cv2.imshow('img0', img0)
        cv2.imshow('img1', img1)
        cv2.waitKey(100)

    # cv2.imshow('img', gray)
    # cv2.waitKey(100)
    ret, mtx0, dist0, mtx1, dist1, R, T, E, F = cv2.stereoCalibrate(objpoints0,
            imgpoints0, imgpoints1, cameraMatrix0, distCoeffs0, cameraMatrix1,
            distCoeffs1, gray0.shape[::-1])

print(ret)
'''
ret =
mtx = camera matrix
dist = distortion coefficient
R0 = R * R1
T0 = R * T1 + T1
'''
print('\n\n\n'.join([str(x) for x in [ret, mtx0, dist0, mtx1, dist1, R, T, E, F]]))
print(j, 'out of', i, 'detected')
cv2.destroyAllWindows()

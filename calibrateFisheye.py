import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cbrow, cbcol = 6, 10
# this is for 7,11 because you have to have inlier points.
# https://stackoverflow.com/questions/31249037/calibrating-webcam-using-python-and-opencv-error/36441746

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)


# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('calb/*.jpg')

# keep track of how many were detected out of the total images looked at
i, j = 0, 0

for fname in images:
    i += 1
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5,5),6,6)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        j += 1

        # image, corners, winSize, zeroZone, criteria
        corners2 = cv2.cornerSubPix(gray,
                                    corners,
                                    (11, 11),
                                    (-1, -1),
                                    criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (cbcol, cbrow), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)

    # cv2.imshow('img', gray)
    # cv2.waitKey(100)

    # print(objpoints, imgpoints, gray.shape[::-1], None, None)

ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpoints,
                                                     imgpoints,
                                                     gray.shape[::-1],
                                                     None, None)
print('\n'.join([ret, mtx, dist, rvecs, tvecs, '']))
print(j, 'out of', i, 'detected')
cv2.destroyAllWindows()

''' calibrateCamera Finds the camera intrinsic and extrinsic parameters from
several views of a calibration pattern.

C++: double calibrateCamera(InputArrayOfArrays objectPoints, InputArrayOfArrays
imagePoints, Size imageSize, InputOutputArray cameraMatrix, InputOutputArray
distCoeffs, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags=0,
TermCriteria criteria=TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 30,
DBL_EPSILON) ) Python: cv2.calibrateCamera(objectPoints, imagePoints,
imageSize[, cameraMatrix[, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]]])
→ retval, cameraMatrix, distCoeffs, rvecs, tvecs C: double
cvCalibrateCamera2(const CvMat* object_points, const CvMat* image_points, const
CvMat* point_counts, CvSize image_size, CvMat* camera_matrix, CvMat*
distortion_coeffs, CvMat* rotation_vectors=NULL, CvMat*
translation_vectors=NULL, int flags=0, CvTermCriteria term_crit=cvTermCriteria(
CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30,DBL_EPSILON) ) Python:
cv.CalibrateCamera2(objectPoints, imagePoints, pointCounts, imageSize,
cameraMatrix, distCoeffs, rvecs, tvecs, flags=0) → None Parameters:
objectPoints – In the new interface it is a vector of vectors of calibration
pattern points in the calibration pattern coordinate space (e.g.
std::vector<std::vector<cv::Vec3f>>). The outer vector contains as many
elements as the number of the pattern views. If the same calibration pattern is
shown in each view and it is fully visible, all the vectors will be the same.
Although, it is possible to use partially occluded patterns, or even different
patterns in different views. Then, the vectors will be different. The points
are 3D, but since they are in a pattern coordinate system, then, if the rig is
planar, it may make sense to put the model to a XY coordinate plane so that
Z-coordinate of each input object point is 0.

In the old interface all the vectors of object points from different views are
concatenated together.

imagePoints – In the new interface it is a vector of vectors of the projections
of calibration pattern points (e.g. std::vector<std::vector<cv::Vec2f>>).
imagePoints.size() and objectPoints.size() and imagePoints[i].size() must be
equal to objectPoints[i].size() for each i.

In the old interface all the vectors of object points from different views are
concatenated together.

point_counts – In the old interface this is a vector of integers, containing as
many elements, as the number of views of the calibration pattern. Each element
is the number of points in each view. Usually, all the elements are the same
and equal to the number of feature points on the calibration pattern.
imageSize – Size of the image used only to initialize the intrinsic camera
matrix.  cameraMatrix – Output 3x3 floating-point camera matrix  A =
\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1} . If
CV_CALIB_USE_INTRINSIC_GUESS and/or CV_CALIB_FIX_ASPECT_RATIO are specified,
some or all of fx, fy, cx, cy must be initialized before calling the function.
distCoeffs – Output vector of distortion coefficients  (k_1, k_2, p_1, p_2[,
k_3[, k_4, k_5, k_6]]) of 4, 5, or 8 elements.  rvecs – Output vector of
rotation vectors (see Rodrigues() ) estimated for each pattern view (e.g.
std::vector<cv::Mat>>). That is, each k-th rotation vector together with the
corresponding k-th translation vector (see the next output parameter
description) brings the calibration pattern from the model coordinate space (in
which object points are specified) to the world coordinate space, that is, a
real position of the calibration pattern in the k-th pattern view (k=0.. M -1).
tvecs – Output vector of translation vectors estimated for each pattern view.
flags – Different flags that may be zero or a combination of the following
values:

CV_CALIB_USE_INTRINSIC_GUESS cameraMatrix contains valid initial values of fx,
fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to
the image center ( imageSize is used), and focal distances are computed in a
least-squares fashion. Note, that if intrinsic parameters are known, there is
no need to use this function just to estimate extrinsic parameters. Use
solvePnP() instead.  CV_CALIB_FIX_PRINCIPAL_POINT The principal point is not
changed during the global optimization. It stays at the center or at a
different location specified when CV_CALIB_USE_INTRINSIC_GUESS is set too.
CV_CALIB_FIX_ASPECT_RATIO The functions considers only fy as a free parameter.
The ratio fx/fy stays the same as in the input cameraMatrix . When
CV_CALIB_USE_INTRINSIC_GUESS is not set, the actual input values of fx and fy
are ignored, only their ratio is computed and used further.
CV_CALIB_ZERO_TANGENT_DIST Tangential distortion coefficients  (p_1, p_2) are
set to zeros and stay zero.  CV_CALIB_FIX_K1,...,CV_CALIB_FIX_K6 The
corresponding radial distortion coefficient is not changed during the
optimization. If CV_CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
CV_CALIB_RATIONAL_MODEL Coefficients k4, k5, and k6 are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make
the calibration function use the rational model and return 8 coefficients. If
the flag is not set, the function computes and returns only 5 distortion
coefficients.  criteria – Termination criteria for the iterative optimization
algorithm.  term_crit – same as criteria.  '''

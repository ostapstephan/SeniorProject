import numpy as np
import cv2
from time import time
import sys

from cameras import cam2mat as cameraMatrix0
from cameras import cam2dcoef as distCoeffs0

from cameras import cam3mat as cameraMatrix1
from cameras import cam3dcoef as distCoeffs1

cameraMatrix0 = np.array(cameraMatrix0)
distCoeffs0 = np.array(distCoeffs0)
cameraMatrix1 = np.array(cameraMatrix1)
distCoeffs1 = np.array(distCoeffs1)

print(cameraMatrix0)
print(cameraMatrix1)
print(distCoeffs0)
print(distCoeffs1)

#these are the measurements of the paper
objectPoints = np.array([(0, 0, 0), (10, 0, 0), (10, -7.5, 0), (0, -7.5,
    0)])*2.54/1.39

def getHomoMat(rot0,t0):
    '''takes in rotation and translation vectors and returns homography matrix to
    go from one camera to another''' 
    a0, _ = cv2.Rodrigues(rot0)
    h0 = np.zeros((4,4)) 
    h0[3,3] = 1
    h0[:3,:3] = a0
    h0[:3,3] = t0.T[0]
    return h0


def solveperp(imagePoints, method):
    if method == 1:
        # print('image points:',imagePoints)
        # obj points (size of led array), image points(pixel values of said
        # points, camera mat(intrinsic camera matrix, distCoef(distortion
        # coefficients of the camera from the intrinsic calibration)
        return cv2.solvePnP(objectPoints, imagePoints, cameraMatrix0,
                distCoeffs0)
    elif method == 2:
        return cv2.solvePnPRansac(
            objectPoints, imagePoints, cameraMatrix0, distCoeffs0
        )
    else:
        return cv2.solveP3P(objectPoints, imagePoints, cameraMatrix0,
                distCoeffs0)

def solveperp1(imagePoints, method):
    if method == 1:
        # print(imagePoints)
        # print(type(imagePoints))
        return cv2.solvePnP(objectPoints, imagePoints, cameraMatrix1,
                distCoeffs1)
    elif method == 2:
        return cv2.solvePnPRansac(
            objectPoints, imagePoints, cameraMatrix1, distCoeffs1
        )
    else:
        return cv2.solveP3P(objectPoints, imagePoints, cameraMatrix1,
                distCoeffs1)


# def cameraPoseFromHomography(H):
    # pose = np.eye(4, 4)
    # norm0 = np.linalg.norm(H[:3,0])  
    # norm1 = np.linalg.norm(H[:3,1])  
    # tnorm = (norm0 + norm1) / 2
    # # p1 = H[:,0]
    # # p2 = pose[:,0]
    # temp = np.array(pose[:3,0])
    # cv2.normalize(H[:3,0], temp)
    # pose[:3,0] = temp

    # # p1 = H.col(1);           // Pointer to second column of H
    # # p2 = pose.col(1);        // Pointer to second column of pose (empty)
    # temp2 = np.array(pose[:3,1])
    # cv2.normalize(H[:3,1], temp2)
    # pose[:3,1] = temp2

    # # p1 = pose.col(0);
    # # p2 = pose.col(1);
    # # Mat p3 = p1.cross(p2);   // Computes the cross-product of p1 and p2
    # # Mat c2 = pose.col(2);    // Pointer to third column of pose
    # # p3.copyTo(c2);       // Third column is the crossproduct of columns one and two
    # pose[:3,2] = np.cross(pose[:3,0],pose[:3,1])

    # pose[:3,3] = H[:3,2]/tnorm #;  //vector t [R|t] is the last column of pose
    # return pose

def cameraPoseFromHomography(H):
    norm1 = np.linalg.norm(H[:, 0])
    norm2 = np.linalg.norm(H[:, 1])
    tnorm = (norm1 + norm2) / 2.0;

    H1 = H[:, 0] / norm1
    H2 = H[:, 1] / norm2
    H3 = np.cross(H1, H2)
    T = H[:, 2] / tnorm

    ret = np.zeros((4,4))
    ret[:3,:] = np.array([H1, H2, H3, T]).transpose()
    ret[3,3] = 1
    return ret




def order_points(pts, img):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect =  [[0,0],[0,0],[0,0],[0,0]]

    if len(pts) != 4:
        return np.array(rect)

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = [sum(pt.pt) for pt in pts]
    # print(type(np.argmin(s)))
    # print(type( pts[int(np.argmin(s) )].pt ))
    rect[0] = pts[int(np.argmin(s))]
    rect[2] = pts[int(np.argmax(s))]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = [pt.pt[0] - pt.pt[1] for pt in pts]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    rect = np.array([(pt.pt[0], pt.pt[1]) for pt in rect])
    shift = 0
    for p in rect:
        # print( img[int(p[1]), int(p[0]) ])
        if img[int(p[1]), int(p[0]) ] == 0:
            break
        # shift += 1
        shift = p
    # print(shift) 
    # rect = rect[shift:] + rect[:shift]

    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [
            [0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ],
        dtype="float32"
    )

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

vc0 = cv2.VideoCapture(0)
ret, frame0 = vc0.read()
if not ret :
    raise Exception

vc1 = cv2.VideoCapture(2)
ret, frame1 = vc1.read()
if not ret :
    raise Exception


cv2.namedWindow("cam0")
cv2.namedWindow("cam1")

for i in range(2):
    vc0.read()
    vc1.read()

print("done with loop")

# vout = None
# if (int(sys.argv[5])):
#    fourcc = cv2.VideoWriter_fourcc(*'x264')
#    vout = cv2.VideoWriter('pupiltest.mp4', fourcc, 24.0,
#    (int(vc.get(3)),int(vc.get(4))))

if vc0.isOpened():  # try to get the first frame
    rval, frame0 = vc0.read()
else:
    rval = False
    print("failed reading the frame cam 0")
    exit()

if vc1.isOpened():  # try to get the first frame
    rval, frame1 = vc1.read()
else:
    print("failed reading the frame cam 1")
    rval = False
    exit()


params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255
# Filter by Area.
params.filterByArea = True
params.minArea = 10
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.7
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.4
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.3
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

ptime = time()
nf = 0



retval0, rvec0, tvec0 = 0,0,0
retval1, rvec1, tvec1 = 0,0,0

while rval:
    # frame    = cv2.rotate(frame, cv2.ROTATE_180)
    roi_gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    roi_gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    keypoints0 = detector.detect(roi_gray0)
    keypoints1 = detector.detect(roi_gray1)
    ii=0 
    for point in keypoints0:
        if ii==0:
            frame0 = cv2.drawMarker(frame0, (int(point.pt[0]),int(point.pt[1])), (0, 0, 255))
        else:
            frame0 = cv2.drawMarker(frame0, (int(point.pt[0]),int(point.pt[1])),(0, 255, 255))
        ii+=1
    ii=0 
    for point in keypoints1:
        if ii==0:
            frame1 = cv2.drawMarker(frame1, (int(point.pt[0]),int(point.pt[1])), (0, 0, 255))
        else:
            frame1 = cv2.drawMarker(frame1, (int(point.pt[0]),int(point.pt[1])),(0, 255, 255))
        ii+=1

    imagePoints0 = order_points(keypoints0, roi_gray0)
    imagePoints1 = order_points(keypoints1, roi_gray1)

    try: 
        retval0, rvec0, tvec0 = solveperp(imagePoints0, 1)
    except:
        # print('didnt solve pnp0',len(keypoints0))
        pass
    try: 
        retval1, rvec1, tvec1 = solveperp1(imagePoints1, 1)
    except:
        # print('didnt solve pnp1',len(keypoints1))
        pass

    # print(rvec0)
    # print(rvec1)
    cv2.imshow("cam0", frame0)
    cv2.imshow("cam1", frame1)
 
    # if vout:
    #    vout.write(frame) 
    nf = nf + 1
    if time() - ptime > 5:
        PointArray0 = []
        PointArray1 = []
        for pointt in keypoints0:
            p = [int(pointt.pt[0]), int(pointt.pt[1])]
            PointArray0.append(p)
        for pointt in keypoints1:
            p = [int(pointt.pt[0]), int(pointt.pt[1])]
            PointArray1.append(p)
        PointArray0 = np.array(PointArray0)
        PointArray1 = np.array(PointArray1)
        
        print('\n R-Vec0\n', rvec0 ,'\n' )
        print('\n T-Vec0\n', tvec0 ,'\n' )
        #print(str(nf / (time() - ptime))) # framerate
        try: 
            a,_ = cv2.Rodrigues(rvec0)
            print(a)
            b,_ = cv2.Rodrigues(rvec1)
            print(b)
            homo0 = getHomoMat(rvec0,tvec0)
            homo1 = getHomoMat(rvec1,tvec1)
            print('homo 0',homo0)
            print('homo 1',homo1)
            h1inv = np.linalg.inv(homo1)
            print('h1inv',h1inv)
            pt0 = keypoints0[0].pt
            pt1 = keypoints1[0].pt
            p0 = np.zeros((4,1))  
            p0[:2] = np.array([pt0]).T
            p0[3]= 1 
            r0 = np.matmul(homo0,p0)
            p1 = np.zeros((4,1))  
            p1[:2] = np.array([pt1]).T
            p1[3]= 1 
            r1 = np.matmul(homo1,p1)
            print(r0-r1)
            print('\n\n\n')
            print(type(keypoints0))
            h01,blah = cv2.findHomography(PointArray0,PointArray1)
            print(h01)
            p01 = cameraPoseFromHomography(h01)
            print(p01)
            print('\n',np.matmul(p01,r0),'\n',r1)
        except Exception as e:
            print(e)
            print("not the right amount of parameters into rodrigues")
            pass
        print('\n R-Vec1\n', rvec1 ,'\n')
        print('\n T-Vec1\n', tvec1 )
 
        ptime = time()
        nf = 0

    key = cv2.waitKey(10) 
    if key & 0xFF == ord('q'):
        break
    elif key == 32:
        cv2.imwrite('data/1testimage.png', frame0)
        cv2.imwrite('data/0testimage.png', frame1)

    rval, frame0 = vc0.read()
    rval, frame1 = vc1.read()

cv2.destroyWindow("cam0")
cv2.destroyWindow("cam1")
vc0.release()
vc1.release()

# if vout:
#    vout.release()

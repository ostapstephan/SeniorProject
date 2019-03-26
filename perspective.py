import numpy as np
import cv2
from time import time
import sys

cameraMatrix = np.array( \
[[526.97673521,   0.,		 310.67820665],
 [  0.,		 523.77758159, 255.58792434],
 [  0.,		   0.,		   1.		]])

objectPoints = np.array([(-7,3.5,0),(0,3.5,0),(0,0,0),(-7,0,0)])

distCoeffs = np.array([[-0.18362732,  0.42600488,  0.00134821, -0.00119089, -0.43325132]])

def solveperp(imagePoints, method):
	if method == 1:
		return cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
	elif method == 2:
		return cv2.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs)
	else:
		return cv2.solveP3P(objectPoints, imagePoints, cameraMatrix, distCoeffs)


def order_points(pts):
	if len(pts) != 4:
		return np.zeros((4,2))
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = [0,0,0,0]
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = [sum(pt.pt) for pt in pts]
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = [pt.pt[0]-pt.pt[1] for pt in pts]
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	rect = np.array([(pt.pt[0],pt.pt[1]) for pt in rect])
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped


cv2.namedWindow("preview")
vc = cv2.VideoCapture(-1)
print(vc.get(3))
print(vc.get(4))
# vout = None
# if (int(sys.argv[5])):
#	 fourcc = cv2.VideoWriter_fourcc(*'x264')
#	 vout = cv2.VideoWriter('pupiltest.mp4', fourcc, 24.0, (int(vc.get(3)),int(vc.get(4))))

if vc.isOpened(): # try to get the first frame
	rval, frame = vc.read()
else:
	rval = False

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255
# Filter by Area.
params.filterByArea = True
params.minArea = 50
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.4
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
while rval:
	frame = cv2.rotate(frame, cv2.ROTATE_180)
	roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	roi_gray = 255-roi_gray

	keypoints = detector.detect(255-roi_gray)
	for point in keypoints:
		frame = cv2.drawMarker(frame, (int(point.pt[0]),int(point.pt[1])), (0,0,255))

	print(keypoints)

	imagePoints = order_points(keypoints)
	retval, rvec, tvec = solveperp(imagePoints, 1)
	print(rvec)
	print(tvec)

	cv2.imshow("preview", frame)
	# if vout:
	#	 vout.write(frame)
	nf = nf + 1
	if time() - ptime > 5:
		PointArray = []
		for pointt in keypoints:
			p = [int(pointt.pt[0]),int(pointt.pt[1])]
			PointArray.append(p)

		print(PointArray)

		print(str(nf/(time()-ptime)))
		ptime = time()
		nf = 0
	key = cv2.waitKey(20)
	if key == 27: # exit on ESC
		break
	elif key == 32:
		cv2.imwrite('testimage.png',frame);
	rval, frame = vc.read()

cv2.destroyWindow("preview")
vc.release()
# if vout:
#	 vout.release()

import cv2

def calibrate(vc):
    cv2.namedWindow("calib")
    # vc = cv2.VideoCapture('http://199.98.27.252:6680/?action=stream')
    # vc = cv2.VideoCapture(2)
    # print(vc.get(3))
    # print(vc.get(4))

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    # face_cascade = cv2.CascadeClassifier('trained/haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier('trained/haarcascade_eye.xml')
    glass_cascade = cv2.CascadeClassifier('trained/haarcascade_eye_tree_eyeglasses.xml')
    # reye_cascade = cv2.CascadeClassifier('trained/haarcascade_righteye_2splits.xml')
    # leye_cascade = cv2.CascadeClassifier('trained/haarcascade_lefteye_2splits.xml')

    maxex = int(vc.get(4)/2)
    maxey = int(vc.get(3)/2)
    minex = maxex
    miney = maxey
    print(maxex)
    while rval:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_color = frame

        eyes = glass_cascade.detectMultiScale(roi_gray, scaleFactor=1.2)
        if len(eyes) == 1:
            (ex,ey,ew,eh) = eyes[0]
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
            maxex = max(maxex,ex+ew)
            maxey = max(maxex,ey+eh)
            minex = min(minex,ex)
            miney = min(miney,ey)

        cv2.rectangle(roi_color,(minex,miney),(maxex,maxey),(255,0,0),2)
        cv2.imshow("calib", roi_color)

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
        rval, frame = vc.read()

    cv2.destroyWindow("calib")
    # vc.release()
    return (minex,miney,maxex,maxey)

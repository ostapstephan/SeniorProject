file = cv2.FileStorage('calib.xml', cv2.FileStorage_READ)
file.getNode('NODE').mat()

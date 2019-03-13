import cv2


def readXML(fileName):
    f = cv2.FileStorage(fileName, cv2.FileStorage_READ)
    ret = []
    for key in f.root().keys():
        node = f.getNode(key)
        if node.isInt() or node.isReal():
            ret += [node.real()]
        elif node.type() == 5:
            ret += [node.mat()]
        else:
            pass
    return ret

import cv2
import requests 
import numpy as np

stream = requests.get('http://192.168.1.251:8080/?action=stream', stream=True)
bytez = b''
while True:
    bytez += stream.raw.read(1024)
    h = bytez.find(b'Timestamp:')
    a = bytez.find(b'\xff\xd8')
    b = bytez.find(b'\xff\xd9')
    if a != -1 and b != -1:
        print(bytez[h+11:a-4])
        jpg = bytez[a:b+2]
        bytez = bytez[b+2:]
        i = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow('i', i)
        if cv2.waitKey(1) == 27:
            exit(0)

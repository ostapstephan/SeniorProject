import cv2
import os
import subprocess as sp
import sys
import numpy as np
import time
import pickle as pk

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from matrix import get_pupil_transformation_matrix
sys.path.append(os.path.abspath('../../TEST'))
sys.path.append(os.path.abspath('../../TEST/shared_modules'))
from pupil_detectors import Detector_3D
from methods import Roi
sys.path.append(os.path.abspath('../'))

from cameras import cam0mat as cameraMatrix0
from cameras import cam0dcoef as distCoeffs0
from cameras import cam1mat as cameraMatrix1
from cameras import cam1dcoef as distCoeffs1

cameraMatrix0 = np.array(cameraMatrix0)
distCoeffs0 = np.array(distCoeffs0)

cameraMatrix1 = np.array(cameraMatrix1)
distCoeffs1 = np.array(distCoeffs1)

def mse(real,predict):
    return(np.mean((real.flatten()-predict.flatten()) **2))


def myerr(real,pred):
    

def genTrainAndVal(d): #split the features and labels of the training data 80:10:10 train, validation and test
    z = d.shape[0]	
    nv = int( z *.1) 	# num validation and test samp 
    return d[:nv], d[nv:nv*2], d[nv*2:]


def lineIntersection(planePoint, planeNormal, linePoint, lineDirection):  #THIS FUNCTION
    if np.dot(planeNormal,lineDirection) == 0:
        return planePoint

    t = (np.dot(planeNormal,planePoint) - np.dot(planeNormal,linePoint)) / np.dot(planeNormal,lineDirection)
    return linePoint + t*lineDirection;

def useHomog(plane,pupil,sphere,R,t):
    H = np.eye(4)
    H[:3,:3] = cv2.Rodrigues(R)[0]
    H[:3,3]= t 
    
    sphere2 = H[:3,:3] @ sphere + H[:3,3] # THESE TWO LINES
    pupil2  = H[:3,:3] @ pupil  + H[:3,3]   # THATS THIS ONE TOO 
    gaze = pupil2-sphere2  
    gazeEnd = lineIntersection(plane[0],np.cross(plane[1]-plane[0],plane[2]-plane[1]), pupil2, gaze) 
    return gazeEnd  #predi
TIMEOUT = 10000
FFMPEG_BIN = "ffmpeg"

with open ("training/databig.pickle", 'rb') as handle: 
    Data = pk.load(handle) 


#rethink how to get the values for the H matrix later
# for i in range(len(Data)): 
    # #pulled outta the other code 
    # plane = Data[i][0]['corners']
    # pupil = Data[i][0]['pupilCenter']['center']
    # sphere= Data[i][0]['eyeCenter']['center'] 
    
    # t = np.array([-0.64, -1.28, 0.0])
    # R = np.array([-14.0,40.0,143])     

    # pred = useHomog(plane,pupil,sphere,R,t)
        
    # actual = Data[i][1]['3dpoint']
    # print(pred,actual)

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


    # plane = Data[i][0]['corners']
    # pupil = Data[i][0]['pupilCenter']['center']
    # sphere= Data[i][0]['eyeCenter']['center']  
    # t = np.array([-0.64, -1.28, 0.0])
    # R = np.array([-14.0,40.0,143])     
    # feat[i][23:] = useHomog(plane,pupil,sphere,R,t) 



    lab[i][:] =np.array( Data[i][1]['2dpoint'] )

    if i ==1:
        print(feat[1],lab[1])

# featt = np.zeros((len(Data), 27 ))
# featt[:,0] = ones.flatten()
# featt[:,1:]= feat
# # print(featt[0])
# feat = featt

dat = np.concatenate((feat,lab),1)
np.random.shuffle(dat)
print(dat.shape)
split = int( len(dat) - .2*len(dat))
print('split = ' , split)
train = dat[:split]
test = dat[split:]

print('train' ,train[:,26:].shape ,train.shape)
print('test' , test.shape)


clf = Ridge(alpha=.1)
clf.fit(train[:,:26],train[:,26:])
pred = clf.predict(test[:,:26])

# Print the Mean squared error
print("Test predictions: ", mse(test[:,26:],pred))




'''
'corners': 
    array([[-297.86904371,  118.11959489,  717.2082578 ],
          [-280.72220576, -226.15180268,  593.62803673],
          [ 257.26313076, -210.56022168,  601.90842078],
          [ 258.23871395,  135.26380944,  738.89282211]]),
   'eyeCenter': {'center': (6.954959367850667,
     8.270334551452644,
     44.68030576516764),
    'radius': 12.0},
   'pupilCenter': {'center': (3.5034175327821435,
     1.7665946344633223,
     35.20464082528444),
     'normal': (-0.2876284862557103, -0.5419783264157768, -0.789638744990266),
     'radius': 1.937747376135334}},
23
  {'3dpoint': array([-293.34530023,  141.49176498, 1413.37953019]),
   '2dpoint': (983.5, 394.5)})

'''

'''
#shuffle data first 
dataBars = shuffle(dataBars)

#split data apart into train val and test
test,val,train = genTrainAndVal(dataBars)

# Pull Data apart into features and labels
testFeat  = np.array(test.drop("MPG",axis=1))
testLab   = np.array(test.MPG)
trainFeat = np.array(train.drop("MPG",axis=1))
trainLab  = np.array(train.MPG)
valFeat   = np.array(val.drop("MPG",axis=1))
valLab    = np.array(val.MPG)

# Perform Linear Regression 
inve = np.linalg.inv(np.matmul(np.transpose(feat),feat)) #the inverse part
trflab = np.transpose(feat) #transpose features times labels
beta_hat = np.matmul(np.matmul(inve,trflab),lab)
pred = np.matmul(feat, beta_hat)
'''



'''
fig, ax = plt.subplots(figsize=[15,10])
negLogAlphas = -np.log10(alphas)
for coef in coefs:
	ax.plot(negLogAlphas, coef.T)

ax.axvline(x=-np.log10(coeffSaved), linestyle="--")
plt.xlabel("-log alpha")
plt.ylabel("coefficients")
plt.title("lasso paths")
#plt.axis('tight')
plt.show()


'''


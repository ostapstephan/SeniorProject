import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import Lasso
from sklearn import linear_model

def genTrainAndVal(d): #split the features and labels of the training data 80:10:10 train, validation and test
	z = d.shape[0]
	nv = int( z *.1) 	# num validation and test samp 
	#print (d[:nv].shape, d[nv:nv*2].shape, d[nv*2:].shape )
	return d[:nv], d[nv:nv*2], d[nv*2:]



data = pd.read_csv("input.csv") #read in the csv into a pandas array
#data.columns = ["variance","skewness","curtosis","entropy","res"]#name the collumns
#dataBars = dataBars.drop("name",axis=1) #Drop the manufacturer name cuz its useless

weights = np.random.uniform(size = len(data.columns.values))
weights[0]=0
print(weights)

#add bias Term 
data.insert(0,"bias",1, allow_duplicates=False)

#shuffle data first 
data = shuffle(data)

#split data apart into train val and test
test,val,train = genTrainAndVal(data)

# Pull Data apart into features and labels
trainFeat = np.array(train.drop("res",axis=1))
trainLab  = np.array(train.res)
valFeat   = np.array(val.drop("res",axis=1))
valLab    = np.array(val.res)
testFeat  = np.array(test.drop("res",axis=1))
testLab   = np.array(test.res)


def Pred(data, w):
	yhat =  np.matmul(data,w)
	return 1/(1+np.exp(-yhat)+1e-7)

print(Pred(trainFeat[0],weights))

def sgd(data,labels, lr, epochs, w):
	# Its stochastic since the data was shuffled 
	likelyhood =[]
	for e in range(epochs):
		sumError = 0
		# loop data rows and update lambda 
		for row in range(len(data)):
			yhat = Pred(data[row],w) #current pred 
			error = labels[row]-yhat #calc an error 
			sumError += error**2 #sqError for showing whether its training
			w = w + lr*error*yhat*(1-yhat)*data[row] #update weights
		likelyhood.append(sum(labels*np.matmul(data,w)-np.log(1+np.exp(np.matmul(data,w)))))
		print('epoch=%d, lr=%.3f, sumError=%.3f' % (e, lr, sumError))
	return w , likelyhood 

def logRegression(train_x,train_y,test_x, lr, epochs, w):
	#Perform SGD with no regularization
	w, l= sgd(train_x,train_y,lr,epochs, w)

	# Genertate Predictions
	yhat = np.round((Pred(test_x,w)), 0)
	return yhat, l 


def sgdL2(data,labels, lr, epochs, w, lam):
	# Its stochastic since the data was shuffled 
	likelyhood =[]
	for e in range(epochs):
		sumError = 0
		# loop data rows and update lambda 
		for row in range(len(data)):
			yhat = Pred(data[row],w) #current pred 
			error = labels[row]-yhat #calc an error 
			sumError += error**2 #sqError for showing whether its training
			w = w + lr*error*yhat*(1-yhat)*data[row] - np.sum(lam*data[row]) #update weights + l2
		likelyhood.append(sum(labels*np.matmul(data,w)-np.log(1+np.exp(np.matmul(data,w)))))
		print('epoch=%d, lr=%.3f, sumError=%.3f' % (e, lr, sumError))
	return w, likelyhood

def logRegressionL2(train_x,train_y,test_x, lr, epochs, w, lam):
	#Perform SGD with L2 regularization
	w, l = sgdL2(train_x,train_y,lr,epochs, w, lam)
	# Genertate Predictions
	yhat = np.round((Pred(test_x,w)), 0)
	return yhat, l 


def accuracy(yhat, y):
	#Compute Accuraccy of Prediction
	numWrong =0
	for val in range(len(y)):
		numWrong += np.abs(y[val]-yhat[val])
	accuracy = float(len(y)-numWrong)/len(y)
	return accuracy


# Input the data and perform the Logistic regression 
pred,lh = logRegression(trainFeat, trainLab, testFeat, .01, 100, weights)
acc1 = accuracy(pred, testLab)



def normData(train, val, test):
	avg = np.average(train,0)
	dev = np.std(train,0)
	dev[0] = 1 #prevent overflow
	#print(avg, dev)
	
	normTrain = (train-avg)/dev
	normVal   = (val  -avg)/dev
	normTest  = (test -avg)/dev
	return(normTrain, normVal, normTest)


nTrain, nVal, nTest = normData(trainFeat, valFeat, testFeat)
weights = np.random.uniform(size = 5)
weights[0]=0

lam = 1e-2
pred, lhwithL2 = logRegressionL2(nTrain, trainLab, nTest, .01, 100, weights, lam)
acc2 = accuracy(pred, testLab)

print("Accuracy Unregularized:",acc1)
print("Accuracy Regularized:",acc2)

ep = np.arange(len(lh))
fig = plt.figure()

plt.plot(ep, lh, "green", label = "Unregularized")
plt.plot(ep, lhwithL2, "red",label = "Regularized")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Likelyhood")
plt.show()

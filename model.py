import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Read Data from CSV files header = none to not ignore first row
frame = pandas.read_csv("train.csv", delim_whitespace=True, header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables

X = dataset[:,2:]
Y = dataset[:,:2]

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)




kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))




# split the data apart for val
train, val = train_test_split(train,test_size=.1)

# Create a Tokenizer instance
#tokenizer = Tokenizer() 
# Only  fit on text this for training data
#tokenizer.fit_on_texts(train.text)

# Create sequenced data for the vocabs
#trainSeq = tokenizer.texts_to_sequences(train.text)
#valSeq	 = tokenizer.texts_to_sequences(val.text)
#testSeq  = tokenizer.texts_to_sequences(test.text)

# Pad the vocabs so that they are all the same len
#lenPad=185
#trainPad= pad_sequences(trainSeq,maxlen=lenPad)
#valPad= pad_sequences(valSeq,maxlen=lenPad)
#testPad= pad_sequences(testSeq,maxlen=lenPad)

# how long to make the embedding vector
#embeddingVectorLen = 32
#vocabLength = len(tokenizer.word_index)

# one hot encode the labels 
#train_cat = np.array(train.cat-1)
#train_cat = train_cat.reshape(train_cat.shape[0],1)
#train_cat = keras.utils.to_categorical(train_cat,num_classes=4)

#val_cat = np.array(val.cat-1)
#val_cat = val_cat.reshape(val_cat.shape[0],1)
#val_cat = keras.utils.to_categorical(val_cat,num_classes=4)

#test_cat = np.array(test.cat-1)
#test_cat = test_cat.reshape(test_cat.shape[0],1)
#test_cat = keras.utils.to_categorical(test_cat,num_classes=4)


model = Sequential()
model.add(Dense(32,input_length=))

#model.add(Embedding(vocabLength+1, embeddingVectorLen, input_length=lenPad))
#model.add(Conv1D(8,32,padding = "same",kernel_regularizer=regularizers.l2()))
#model.add(Conv1D(128,32,padding = "same",dilation_rate=3,kernel_regularizer=regularizers.l2()))

model.add(Dropout(.7))
model.add(Flatten())
model.add(Dense(4,activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy,
	optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

model.fit(trainPad, train_cat,
	batch_size=64,
	epochs = 2,
	verbose = 1,
	validation_data =(valPad,val_cat))

score = model.evaluate(testPad,test_cat,verbose= 1)
print("Test loss:",score[0])
print("Test accuracy:",score[1])

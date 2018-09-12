import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, rmsprop
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import CSVLogger
from keras import regularizers
from keras import metrics
from keras.models import load_model
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_recall_fscore_support
import random as ra
from sklearn_pandas import DataFrameMapper

#create dict of diagnosis and category number
def dx_dict(df):
	c = df.Dx.unique()
	r = len(c)
	dx = dict(zip(range(r),c))
	return c,r,dx

#convert diagnosis to float 0-4 in dx column
def sep_diag(df,c):
	r = len(c)
	for i in range(r):
		x = c[i]
		df.loc[df.Dx==x,'dx']=i
	return df
#scales continuous variables to 0 - 1 sigmoid range
def map_scaler(df,m=0,n=1):
	cols = df.columns[3:-1]
#create min max scaler instance for data
	map = [([c],MinMaxScaler(feature_range = (m,n),copy=False)) for c in cols]
#slice out MEG data
	dd = df.loc[:,cols]
#apply map using DataFrameMapper
	mapper=DataFrameMapper(map)
	dm = mapper.fit_transform(dd).astype(np.float32)
#convert target to categorical
	y = np.array([int(x) for x in df.dx]).reshape(-1,1)
	y = np_utils.to_categorical(y,num_classes = 5)
	return dm,y
#create train test split
def split_sample(dm, y, seed, s=.1):
	x_train, x_test, y_train, y_test = train_test_split(dm, y, test_size = s, random_state = seed, stratify = y)
	return x_train, x_test, y_train, y_test
#create categorical keras model
def mlp_model(x_train,y_train):
	dim = x_train.shape[1]
	model = Sequential()
	model.add(Dense(len(y_train), activation='relu', input_dim=dim, kernel_initializer='normal', bias_initializer='zeros'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(200, activation='relu', kernel_initializer='normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(200, activation='relu', kernel_initializer='normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.6))
	model.add(Dense(200, activation='relu', kernel_initializer='normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.6))
	model.add(Dense(5, activation='softmax'))
	#early_stopping_monitor = EarlyStopping(patience = 3)
	adam = keras.optimizers.Adam(lr=.0001, decay = 0.01)
	#sgd = keras.optimizers.SGD()
	model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['categorical_accuracy'])
	return model
#fit model and compute accuracy for each epoch
def fit_model(x_train,x_test,y_train,y_test,n=500, name='all_diagnosis'):
	model = mlp_model(x_train,y_train)
#batch size is all data
	n_batch = x_train.shape[0]
#record accuracy for each epoch
	epoch_log = CSVLogger(name + '_accuracy.csv')
	callbacks = [epoch_log]
	model.fit(x_train,y_train,epochs=n,verbose= True,batch_size=n_batch, callbacks=callbacks)
	#score = model.evaluate(x_test, y_test, batch_size=n_batch)
	y_pred = model.predict(x_test)
	#model.save(name+"_model.h5")
	del model
	return y_pred

def compute_scores(y_true, y_pred, name = 'all_diagnosis'):
	y_true = list(np.argmax(y_test,axis=1))
	y_pred = list(np.argmax(y_pred,axis=1))
	tp,fp,fn,tn,a = 0,0,0,0,0
	a=sum([1 for i in range(len(y_true)) if y_pred[i]==y_true[i]])
	tn=sum([1 for i in range(len(y_true)) if y_true[i]==0 and y_pred[i]==0])
	tp=sum([1 for i in range(len(y_true)) if y_pred[i]==y_true[i] & y_true[i]>0])
	fp=sum([1 for i in range(len(y_true)) if y_true[i]==0 and y_pred[i]>0])
	fn=sum([1 for i in range(len(y_true)) if y_true[i]>0 and y_pred[i]==0])
	acc = round(float(a)/float(len(y_true)),3)
	tpr=round(float(tp)/float(tp+fn),3)
	tnr=round(float(tn)/float(tn+fp),3)
	ppv=round(float(tp)/float(tp+fp),3)
	npv=round(float(tn)/float(tn+fn),3)
	f1=round(float(2*tp)/float(2*tp+fp+fn),3)
	cols = ['Accuracy','TPR','TNR','PPV','NPV','F1']
	score = [acc,tpr,tnr,ppv,npv,f1]
	scores=pd.DataFrame()
	for i in range(len(cols)):
		c = cols[i]
		v = score[i]
		scores.loc[0,c]=v
	scores.to_csv(name + '_scores.csv', index = False)
	return scores


if __name__ == '__main__':
	#load data
	df = pd.read_csv('MEG_sample.csv')
	#set seed
	seed = 73
	ra.seed = 73
	c,r,diagnosis = dx_dict(df)
	df = sep_diag(df,c)
	dm,y = map_scaler(df,m=0,n=1)
	x_train,x_test,y_train,y_test = split_sample(dm, y, seed, s=.1)
	y_pred=fit_model(x_train,x_test,y_train,y_test)
 	scores=compute_scores(y_test,y_pred, name = 'all_diagnosis')

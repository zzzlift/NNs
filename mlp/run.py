import numpy as np
from mlp import *

def loadDataSet(filepath):
	f=gzip.open(filepath,'rb')
	#self.train_set,self.valid_set,self.test_set=cPickle.load(f)
	data=cPickle.load(f)
	f.close()
	return data


	
if __name__=="__main__":
	mlp=mlp()
	
	mlp.hidden_neurons=20
	mlp.output_neurons=10
	mlp.max_iteration=1000
	mlp.error_goal=0.01
	mlp.batch=10
	
	data=loadDataSet('../data/mnist.pkl.gz')
	
	#because the sum of errors in every batch as one update, if you use 10000 samples in a batch, the sum error too large, and it will casue the activation function full,cause the output keep the same time, because the output neurons are full. so the output value is -1 or 1.so y(1-y)=0 or -2, so the gradient is 0 or -2 always, finally the gradient is 0 and cause can't update the weigth
	train_samples=10000
	x=data[0][0][0:train_samples]
	y=data[0][1][0:train_samples]
	#sigmoid needs the [-1,1] range, not the [0,1]
	x=mlp.normalize(x)
	mlp.train_set=x
	#mlp.labels=np.matrix(y).T
	mlp.labels=mlp.labelConvert(y,10)
	
	test_samples=100
	x_test=data[2][0][0:test_samples]
	y_test=data[2][1][0:test_samples]
	x_test=mlp.normalize(x_test)
	mlp.test_set=x_test
	mlp.test_labels=mlp.labelConvert(y_test,10)
	
	#print mlp.labels.shape
	#print mlp.train_set
	#print mlp.train_set[0].shape
	'''
	mlp.train_set[0]=mlp.normalize(mlp.train_set[0])
	mlp.test_set[0]=mlp.normalize(mlp.test_set[0])
	'''
	
	mlp.bpTrain()
	
	prediction=mlp.BPClassifier()
	
	dis=prediction-mlp.test_labels
	'''
	mse=mlp.errorFunc(dis)
	print mse
	print prediction
	'''
	predict=np.argmax(prediction,axis=1)
	print predict
	dist=predict-np.matrix(y_test).T
	print dist
	
	
	
	
	
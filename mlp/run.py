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
	mlp.hidden_neurons=10
	mlp.output_neurons=10
	mlp.max_iteration=10000
	
	data=loadDataSet('../data/mnist.pkl.gz')
	
	
	number=1000
	x=data[0][0][0:number]
	y=data[0][1][0:number]
	x=mlp.normalize(x)
	mlp.train_set=x
	#mlp.labels=np.matrix(y).T
	mlp.labels=mlp.labelConvert(y,10)
	
	
	x_test=data[2][0][0:number]
	y_test=data[2][1][0:number]
	#x_test=mlp.normalize(x_test)
	mlp.test_set=x_test
	mlp.test_labels=mlp.labelConvert(y_test,10)
	
	print mlp.labels.shape
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
	dist=predict-y_test
	print dist
	
	
	
	
	
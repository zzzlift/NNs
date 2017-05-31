import numpy as np
from mlp import *

def loadDataSet(filepath):
	f=gzip.open(filepath,'rb')
	#self.train_set,self.valid_set,self.test_set=cPickle.load(f)
	data=cPickle.load(f)
	f.close()
	return data
'''
mlp has lots of things to consider. you need to use the momentum factor or inertial factor to search the optimal result
you need to think how many samples in a batch, you can use small number of them, even one sample in a batch, but you can't use too much in a batch, Batch means you add the errors from all samples and upate the weigth once according to the total error. you use small number batch only means you will converge very slow or get a loca optimum. but it won't influence too much. but you use large number batch, you will crash. because every time you update the delta_weigth too big. it just like you use a large step. the large step length. training or learning process in fact is a optimization process. if you can find a very easy way to search the parameter of a function. you don't need neural network at all. you just use math to solve it.
'''

	
if __name__=="__main__":
	mlp=mlp()
	
	mlp.hidden_neurons=20
	mlp.output_neurons=10
	#this is the most important parameter, you have no need to care about the error_goal
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
	#print predict
	dist=predict-np.matrix(y_test).T
	print dist
	
	
	
	
	
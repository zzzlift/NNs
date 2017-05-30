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
	data=loadDataSet('../data/mnist.pkl.gz')
	x=data[0][0]
	y=data[0][1][0:99]
	l=np.empty((0,10))
	for i in xrange(y.shape[0]):
		t=np.zeros((1,10))
		t[0][y[i]]=1
		l=mlp.addRow(l,t)
	print l.shape
	mlp.train_set=x
	#mlp.labels=np.matrix(y).T
	mlp.labels=l
	print mlp.labels.shape
	#print mlp.train_set
	#print mlp.train_set[0].shape
	'''
	mlp.train_set[0]=mlp.normalize(mlp.train_set[0])
	mlp.test_set[0]=mlp.normalize(mlp.test_set[0])
	'''
	
	mlp.bpTrain()
	
	mlp.BPClassifier()
	
	print'dd'
	
	
	
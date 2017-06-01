import numpy as np
import gzip
import cPickle

def normalize(dataMat):
	'''
	[[sample1],
	[sample2],
	[sample3]]
	'''
	[m,n]=dataMat.shape
	for i in xrange(n-1):
		'''
		use standard deviation is better than the max-min, becasue it considers the distribution.
		if big values are not too many, most scatter in a small area. the deviation will be small
		it won't project most data into a too small area.
		dataMat[:,i]-mean is a every element in the vector minus a same value
		'''
	dataMat[:,i]=(dataMat[:,i]-np.mean(dataMat[:,i]))/(np.std(dataMat[:,i])+1.0e-10)
	return dataMat
	
def labelConvert(y,types):
	l=np.empty((0,types))
	for i in xrange(y.shape[0]):
		t=np.zeros((1,10))
		t[0][y[i]]=1
		l=addRow(l,t)
	#print l.shape
	return l
	
def loadDataSetgzip(filepath):
	f=gzip.open(filepath,'rb')
	#self.train_set,self.valid_set,self.test_set=cPickle.load(f)
	data=cPickle.load(f)
	f.close()
	return data
	
def addRow(mat1,mat2):
	mat=np.vstack((mat1,mat2))
	return mat
	
def addCol(mat1,mat2):
	mat=np.hstack((mat1,mat2))
	return mat
import numpy as np
from mlp import *

if __name__=="__main__":
	mlp=mlp()
	mlp.loadDataSet('../data/mnist.pkl.gz')
	#print mlp.train_set
	print mlp.train_set[0].shape
	'''
	mlp.train_set[0]=mlp.normalize(mlp.train_set[0])
	mlp.test_set[0]=mlp.normalize(mlp.test_set[0])
	'''
	
	mlp.bpTrain()
	
	mlp.BPClassifier()
	
	print'dd'
	
	
	
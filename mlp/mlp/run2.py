import numpy as np
from layer import *
from props import *
from utils import *
from network import *

if __name__=="__main__":
	network=network()
	network.max_itration=1000
	network.batch=15
	network.props=props()
	data=loadDataSetgzip('../../data/mnist.pkl.gz')
	
	train_samples=10000
	xAll=data[0][0][0:train_samples]
	yAll=data[0][1][0:train_samples]
	
	network.labels=labelConvert(yAll,10)
	network.xAll=xAll
	network.train()
	
	test_samples=100
	network.test_set=data[2][0][0:test_samples]
	y_test=np.matrix(data[2][1][0:test_samples]).T
	#print network.test_set.shape
	predict=network.predict()
	y_hat=np.argmax(np.matrix(predict),axis=1)
	dist=y_test-y_hat
	print dist
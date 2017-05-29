import gzip
import numpy as np

def __init__(self):
	f=gzip.open(dataset)
	
def init_hiddenWB(self):
	'''
	self_hidden_w=
	[neuron1:[w1,w2...],
	neuron2:[w1,....],
	neuron3:[w2,...]]
	self_hidden_bias=[neuron1,neuron2,neuron3....]
	'''
	self.hidden_w = 2*np.random.random((self.hidden_neurons,self.input)) - 1
	self.hidden_bias=2*np.random.random((self.hidden_neurons,1)) - 1
	self.hidden_wb=addRow(self.hidden_w,self.hidden_bias)
	
	

def init_outputWB(self):
	self.output_w=2*np.random.random((self.ouput_neurons,self.hidden_neurons)) - 1;
	self.output_bias=2*np.random.random((self.ouput_neurons,1)) - 1
	self.output_wb=addRow(self.output_w,self.output_bias)


def loadDataSet(self,filename):
	f=gzip.open('mnist.pkl.gz','rb')
	self.train_set,self.valid_set,self.test_set=cPickle.load(f)
	f.close()
	
def normalize(self,dataMat):
	
	
def addRow(self,mat1,mat2):
	mat=np.hstack((mat1,mat2))
	return mat
	
def addCol(self,mat1,mat2):
	mat=np.hstack((mat1,mat2))
	return mat

def drawClassScatter(self,plt):
	
def bpTrain(self):
	x=self.train_set[0]
	y=self.train_set[1]
	self.init_hiddenWB()
	self.init_outputWB()
	
	for i in xrange(self.itration):
		self.hi
	
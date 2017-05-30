import gzip
import numpy as np

def __init__(self):
	'''
	parameter settings
	error_goal is the threshold of error sensitivity
	max_iteration is the max iteration
	iteration is the current iteration when finish
	hidden_neurons is the number of hidden neurons
	eta is the learning rate/step length
	mc is the momentum parameter
	'''
	self.error_goal=0.01
	self.max_iteration=1000
	self.iteration=0
	self.eta=0.1
	self.mc=0.3
	self.hidden_neurons=4
	self.output_neurons=1
	'''
	automatic parameter
	errorList is the current error for visulation
	train_set is the training set
	valid_set is for overfitting
	test_set is for test
	inputDim is the dimension of samples
	train_sampleNum is the number of training samples
	'''
	
	self.errorList=[]
	self.train_set=0
	self.valid_set=0
	self.test_set=0
	self.inputDim=0
	self.train_sampleNum=0
	
def init_hiddenWB(self):
	'''
	self_hidden_w=
	[neuron1:[w1,w2...],
	neuron2:[w1,....],
	neuron3:[w2,...]]
	self_hidden_bias=[neuron1,neuron2,neuron3....]
	'''
	self.hidden_w = 2*np.random.random((self.hidden_neurons,self.inputDim)) - 1
	self.hidden_bias=2*np.random.random((self.hidden_neurons,1)) - 1
	self.hidden_wb=addRow(self.hidden_w,self.hidden_bias)
	
	

def init_outputWB(self):
	self.output_w=2*np.random.random((self.ouput_neurons,self.hidden_neurons)) - 1;
	self.output_bias=2*np.random.random((self.output_neurons,1)) - 1
	self.output_wb=addRow(self.output_w,self.output_bias)


def loadDataSet(self,filename):
	f=gzip.open('mnist.pkl.gz','rb')
	self.train_set,self.valid_set,self.test_set=cPickle.load(f)
	f.close()
	
def normalize(self,dataMat):
	'''
	[[sample1],
	[sample2],
	[sample3]]
	'''
	[m,n]=shape(dataMat)
	for i in xrange(n-1):
		'''
		use standard deviation is better than the max-min, becasue it considers the distribution.
		if big values are not too many, most scatter in a small area. the deviation will be small
		it won't project most data into a too small area.
		dataMat[:,i]-mean is a every element in the vector minus a same value
		'''
		dataMat[:,i]=(dataMat[:,i]-np.mean(dataMat[dataMat[:,i]]))/(np.std(dataMat[:,i]+1.0e-10))
	return dataMat
	
def addRow(self,mat1,mat2):
	mat=np.hstack((mat1,mat2))
	return mat
	
def addCol(self,mat1,mat2):
	mat=np.hstack((mat1,mat2))
	return mat

def drawClassScatter(self,plt):
	
def activation(self,data):
	return 1.0/(1.0+exp(-data))

def errorFunc(self,data):
	return np.sum(np.power(data,2))*0.5

def derivative(self,data):
	return data*(1-data)
	
def bpTrain(self):
	xo=self.train_set[0]
	#train set need add one column for bias
	self.train_sampleNum=self.train_set.shape[0]
	self.inputDim=self.train_set.shape[0]
	x=addCol(xo,np.ones((self.train_sampleNum,1))
	y=self.train_set[1]
	
	self.init_hiddenWB()
	self.init_outputWB()
	
	delat_output_wb_old=0
	delta_hidden_wb_old=0
	
	for i in xrange(self.max_itration):
		'''
		forward process
		'''
		#self.hidden has one more column than the neuron number, due to bias
		hidden_input=np.dot(self.hidden_wb*x)
		#the output dimension is the neuron number, becase the hidden_wb has rows of neuron number, bias is added to every neuron
		hidden_ouput=self.activation(hidden_input)
		hidden_output_extend=self.addCol(hidden_output.T,np.ones((self.train_sampleNum,1))).T
		
		output_input=np.dot(self.output_wb,hidden_output_extend)
		output_ouput=self.activation(output_input)
		
		'''
		backward process
		'''
		error=y-output_output
		mse=self.errorFunc(error)
		self.errorList.append(mse)
		if mse<=self.error_goal:
			self.iteration=i+1
			break;
		
		#error is derivated by a vector, which is a vector
		output_gradient=error*self.derivative(output_output)
		#think about the meaning of every equation, this focus every on one hidden neuron
		delta_output_wb=np.dot(output_gradient,hidden_output_extend.T)
		'''
		update one w_ij, [delta_1,delta_2,...,delta_k]*[w_ji,w_j2,...,w_jk]* (gradient of O_j)*O_i
		update all w_ij, [delta_1,delta_2,...,delta_k]*[[w_1i,w_12,...,w_1k],[w_2i,w_22,...,w_2k],...]*[gradient_1,gradient_2,....]*[O_1,O_2,....]
		because the bias neuron doesn't connect to the previsous layer, so we don't transfer its gradient,we just update its weight in the output layer
		when you use *, one of the element must be vector or scala,if 2 matrix, we must use the np.dot
		'''
		hidden_delta=self.output_wb[:,:-1].T*output_gradient*self.derivative(hidden_output)
		delta_hidden_wb=np.dot(hidden_delta,x.T)
		
		if i==0:
			self.output_wb=self.output_wb+self.eta*delta_output_wb
			self.hidden_wb=self.hidden_wb+self.eta*delta_hidden_wb
		else:
			self.output_wb=self.output_wb+(1.0-self.mc)*self.eta*delta_output_wb + self.mc*delta_output_wb_old
			self.hidden_wb=self.hidden_wb+(1.0-self.mc)*self.eta*delta_hidden_wb + self.mc*delta_hidden_wb_old
		
		delta_output_wb_old=delta_output_wb
		delta_hidden_wb_old=delta_hidden_wb
		
def BPClassifier(self):
	x0=self.test_set[0]
	sampleNum=self.test_set.shape[0]
	x=addCol(x0,np.ones((sampleNum,1))
	y=self.test_set[1]
	hidden_input=np.dot(self.hidden_wb,x.T)
	hidden_output=self.activation(hidden_input)
	row,col=hidden_ouput.shape
	hidden_output_extend=addCol(hidden_output,np.ones(sampleNum,1))
	output_input=np.dot(output_wb,hidden_output_extend.T)
	output=self.activation(output_input)
	

	
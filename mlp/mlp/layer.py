import gzip
import numpy as np
import cPickle
'''
created on June 1 2017
@author: Zack
'''
class layer(object):
	'''
	def __init__(self):
	def init_hiddenWB(self):
	def init_outputWB(self):
	def loadDataSet(self,filepath):
	def normalize(self,dataMat):
	def addRow(self,mat1,mat2):
	def addCol(self,mat1,mat2):
	def activation(self,data):
	def errorFunc(self,data):
	def derivative(self,data):
	def bpTrain(self):
	def BPClassifier(self):
	'''

	'''
	because it use some from *** import * so we use self.function() to identify we invoke this function in this file
	'''
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
		self.max_itration=1000
		self.iteration=0
		self.eta=0.001
		self.mc=0.3
		self.neurons=4
		#receive a class
		self.activation=0
		self.input=0
		self.input_extend=0
		self.inputDim=0
		self.error=100
		self.layer_output=0
		self.first_flag1=True
		self.first_flag2=True
		'''
		automatic parameter
		errorList is the current error for visulation
		train_set is the training set
		labels is the training label
		valid_set is for overfitting
		test_set is for test
		inputDim is the dimension of samples
		train_sampleNum is the number of training samples
		'''
		
		self.valid_set=0
		self.labels=0
		self.inputDim=0
		self.train_sampleNum=0
		self.layer_wb=0
		self.delta_layer_wb_old=0
		
	
	
	
	def init_WB(self):
		self.layer_w=2*np.random.random((self.neurons,self.inputDim)) - 1;
		self.layer_bias=2*np.random.random((self.neurons,1)) - 1
		self.layer_wb=self.addCol(self.layer_w,self.layer_bias)
		
	def addRow(self,mat1,mat2):
		mat=np.vstack((mat1,mat2))
		return mat
		
	def addCol(self,mat1,mat2):
		mat=np.hstack((mat1,mat2))
		return mat
		
	
	def backword(self):
		
		#backword	
		layer_gradient=np.multiply(self.error,self.activation.derivative(self.layer_output))
		delta_layer_wb=np.dot(layer_gradient.T,self.input_extend)
		L1_error=np.dot(layer_gradient,self.layer_wb[:,:-1])

		#update
		if self.first_flag2==True:
			self.layer_wb=self.layer_wb+self.eta*delta_layer_wb
			self.first_flag2=False
		else:
			self.layer_wb=self.layer_wb+(1.0-self.mc)*self.eta*delta_layer_wb + self.mc*self.delta_layer_wb_old
		self.delta_layer_wb_old=delta_layer_wb
		return L1_error
	
	#classifier is the forward process
	def forward(self):
		if self.first_flag1==True:
			self.init_WB()
			self.first_flag1=False
		samples=self.input.shape[0]
		self.input_extend=self.addCol(self.input,np.ones((samples,1)))
		layer_input=np.dot(self.input_extend,self.layer_wb.T)
		layer_output=self.activation.activation(layer_input)
		return layer_output
		

		
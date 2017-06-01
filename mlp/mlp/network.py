import gzip
import numpy as np
import cPickle
'''
created on May 28 2017
@author: Zack
'''
class network(object):
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
		self.hidden_neurons=4
		self.output_neurons=10
		self.batch=10
		#receive a class
		self.activation=0
		self.hidden_output=0
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
		
		self.errorList=[]
		self.train_set=0
		self.valid_set=0
		self.labels=0
		self.test_set=0
		self.test_labels=0
		self.inputDim=0
		self.train_sampleNum=0
	
	
	
	def init_WB(self):
		self.output_w=2*np.random.random((self.output_neurons,self.hidden_neurons)) - 1;
		self.output_bias=2*np.random.random((self.output_neurons,1)) - 1
		self.output_wb=self.addCol(self.output_w,self.output_bias)
		
	def addRow(self,mat1,mat2):
		mat=np.vstack((mat1,mat2))
		return mat
		
	def addCol(self,mat1,mat2):
		mat=np.hstack((mat1,mat2))
		return mat
		
	
	def bpTrain(self):
		#return
		
		self.init_WB()
		delta_output_wb_old=0
		
		
		#forward process
		hidden_output_extend=self.addCol(self.hidden_output,np.ones((self.batch,1)))
		output_input=np.dot(hidden_output_extend,self.output_wb.T)
		output_output=self.activation.activation(output_input)
		
		error=y-output_output
		mse=self.errorFunc(error)
		
		print 'mse '+str(mse)
		self.errorList.append(mse)
		if mse<=self.error_goal:
			self.iteration=i+1
			break;
			
			
		#backword	
		output_gradient=np.multiply(error,self.derivative.derivative(output_output))
		delta_output_wb=np.dot(output_gradient.T,hidden_output_extend)
		hidden_delta=np.multiply(np.dot(output_gradient,self.output_wb[:,:-1]),self.derivative(hidden_output))

		#update
		self.output_wb=self.output_wb+(1.0-self.mc)*self.eta*delta_output_wb + self.mc*delta_output_wb_old
		delta_output_wb_old=delta_output_wb
	
	#classifier is the forward process
	def prediction(self):
		hidden_output_extend=self.addCol(self.hidden_output,np.ones((self.batch,1)))
		output_input=np.dot(hidden_output_extend,self.output_wb.T)
		output_output=self.activation.activation(output_input)
		return output_output
		

		
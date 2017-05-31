import gzip
import numpy as np
import cPickle

class mlp(object):
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
	
	
	
	def labelConvert(self,y,types):
		l=np.empty((0,types))
		for i in xrange(y.shape[0]):
			t=np.zeros((1,10))
			t[0][y[i]]=1
			l=self.addRow(l,t)
		print l.shape
		return l
	
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
		self.hidden_wb=self.addCol(self.hidden_w,self.hidden_bias)
		
		

	def init_outputWB(self):
		self.output_w=2*np.random.random((self.output_neurons,self.hidden_neurons)) - 1;
		self.output_bias=2*np.random.random((self.output_neurons,1)) - 1
		self.output_wb=self.addCol(self.output_w,self.output_bias)


	
		
	def normalize(self,dataMat):
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
		
	def addRow(self,mat1,mat2):
		mat=np.vstack((mat1,mat2))
		return mat
		
	def addCol(self,mat1,mat2):
		mat=np.hstack((mat1,mat2))
		return mat

	#def drawClassScatter(self,plt):
	
	#drawbackness this function max value only can be 1 when the label is 9,8 big value, all the prediction will be 1 because the least is 1 in my samples
	def activation(self,data):
		return 1.0/(1.0+np.exp(-data))

	def errorFunc(self,data):
		return np.sum(np.power(data,2))*0.5

	def derivative(self,data):
		return np.multiply(data,(1-data))
		
	
	def bpTrain(self):
		#return
		x0=self.train_set
		'''
		just take a little piece
		'''
		#x0=x0[0:99,:]
		label_matrix=np.matrix(self.labels)
		y=label_matrix
		print 'labels.shape' + str(self.labels.shape)
		#y[:,1]=labels[0:99]
		#train set need add one column for bias
		self.train_sampleNum=x0.shape[0]
		self.inputDim=x0.shape[1]
		#print self.train_sampleNum
		x=self.addCol(x0,np.ones((self.train_sampleNum,1)))
		
		
		self.init_hiddenWB()
		self.init_outputWB()
		
		delta_output_wb_old=0
		delta_hidden_wb_old=0
		
		for i in xrange(self.max_itration):
			print 'new iteration **************'
			'''
			forward process
			'''
			#self.hidden has one more column than the neuron number, due to bias
			'''
			print self.hidden_wb.shape
			print x.shape
			'''
			hidden_input=np.dot(x,self.hidden_wb.T)
			#print 'hidden_input.shape is '+str(hidden_input.shape)
			#the output dimension is the neuron number, becase the hidden_wb has rows of neuron number, bias is added to every neuron
			hidden_output=self.activation(hidden_input)
			
			#print 'hidden_output.shape is '+str(hidden_output.shape)
			hidden_output_extend=self.addCol(hidden_output,np.ones((self.train_sampleNum,1)))
			#print 'hidden_output_extend.shape is '+str(hidden_output_extend.shape)
			print 'output_wb.shape is '+str(self.output_wb.shape)
			output_input=np.dot(hidden_output_extend,self.output_wb.T)
			#print output_input
			return
			'''
			you know the input dimension should be 2, so the output_input is a matrix with 2 column
			matrix is a good friend, because if only the row number is correct
			your program is correct
			'''
			print 'output_input.shape is '+str(output_input.shape)
			output_output=self.activation(output_input)
			#print output_input
			'''
			backward process
			'''
			print 'y.shape is ' +str(y.shape)
			print 'output_output.shape is '+str(output_output.shape)
			error=y-output_output
			#print output_output
			#sum all elements in a matrix-> scalar
			mse=self.errorFunc(error)
			print error
			self.errorList.append(mse)
			if mse<=self.error_goal:
				self.iteration=i+1
				break;
			
			#error is derivated by a vector, which is a vector
			print 'error.shape is ' + str(error.shape)
			#print self.derivative(output_output).shape
			#because the error is a matrix not a array so you canot use matrix * array
			#print error
			#here output_output all is one so the derivative output*(1-output)=0 always, it cause by output_input so big more than 50 each element
			output_gradient=np.multiply(error,self.derivative(output_output))
			#print output_output
			#think about the meaning of every equation, this focus every on one hidden neuron
			#print 'output_gradient.shape is '+str(output_gradient.shape)
			#print 'hidden_output_extend.shape is '+str(hidden_output_extend.shape)
			#the dimension of delta_output is the output_wb, because each weight should be updated
			
			delta_output_wb=np.dot(output_gradient.T,hidden_output_extend)
			print 'delta_output_wb.shape is '+str(delta_output_wb.shape)
			#print output_gradient
			'''
			update one w_ij, [delta_1,delta_2,...,delta_k]*[w_ji,w_j2,...,w_jk]* (gradient of O_j)*O_i
			update all w_ij, [delta_1,delta_2,...,delta_k]*[[w_1i,w_12,...,w_1k],[w_2i,w_22,...,w_2k],...]*[gradient_1,gradient_2,....]*[O_1,O_2,....]
			because the bias neuron doesn't connect to the previsous layer, so we don't transfer its gradient,we just update its weight in the output layer
			when you use *, one of the element must be vector or scala,if 2 matrix, we must use the np.dot
			'''
			#as for each hidden neuron, because each neuron only can has one gradient, so a samples*hidden_neurons matrix
			#print 'self.output_wb[:,:-1].shape is '+str(self.output_wb[:,:-1].shape)
			#print 'derivative(hidden_output).shape is '+str(self.derivative(hidden_output).shape)
			#hidden_delta should be neurons*1 matrxi
			#only the derivative is mulitply others is dot product
			#np.dot(self.output_wb[:,:-1].T,output_gradient) is the L1_error, just like the L2_error which we use to * the derivative of output_output
			hidden_delta=np.multiply(np.dot(output_gradient,self.output_wb[:,:-1]),self.derivative(hidden_output))
			#print 'hidden_delta.shape is '+str(hidden_delta.shape)
			#print 'x.shape is '+str(x.shape)
			delta_hidden_wb=np.dot(hidden_delta.T,x)
			#print 'delta_hidden_wb.shape is '+str(delta_hidden_wb.shape)
			
			if i==0:
				self.output_wb=self.output_wb+self.eta*delta_output_wb
				self.hidden_wb=self.hidden_wb+self.eta*delta_hidden_wb
				
			else:
				self.output_wb=self.output_wb+(1.0-self.mc)*self.eta*delta_output_wb + self.mc*delta_output_wb_old
				self.hidden_wb=self.hidden_wb+(1.0-self.mc)*self.eta*delta_hidden_wb + self.mc*delta_hidden_wb_old
				#print 'self.output_wb is '+str(self.output_wb)
			#print error	
			#return
			
			delta_output_wb_old=delta_output_wb
			delta_hidden_wb_old=delta_hidden_wb
	
	#classifier is the forward process
	def BPClassifier(self):
		x0=self.test_set
		sampleNum=x0.shape[0]
		x=self.addCol(x0,np.ones((sampleNum,1)))
		label_matrix=np.matrix(self.test_labels)
		y=label_matrix
		hidden_input=np.dot(x,self.hidden_wb.T)
		hidden_output=self.activation(hidden_input)
		hidden_output_extend=self.addCol(hidden_output,np.ones((sampleNum,1)))
		output_input=np.dot(hidden_output_extend,self.output_wb.T)
		output_output=self.activation(output_input)
		return output_output
		

		
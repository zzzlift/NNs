import numpy as np
from layer import *
from props import *
from utils import *


	
class network():

	def __init__(self):
		self.errorList=[]
		self.iteration=0
		self.error_goal=0.01
		self.max_itration=100
		self.batch=10
		self.xAll=0
		self.labels=0
		self.props=0
		self.hidden_layer1=0
		self.hidden_layer2=0
		self.hidden_layer3=0
		self.hidden_layer4=0
		self.output_layer=0
		self.test_set=0
	
	def train(self):
		#sigmoid needs the [-1,1] range, not the [0,1]
		self.xAll=normalize(self.xAll)
		
		#mlp.labels=np.matrix(y).T
		
		
		#initialize the network
		hidden_layer1=layer()
		hidden_layer2=layer()
		output_layer=layer()
		
		
		train_samples=self.xAll.shape[0]
		hidden_layer1.inputDim=self.xAll.shape[1]
		hidden_layer1.neurons=10
		hidden_layer1.activation=self.props
		
		hidden_layer2.inputDim=hidden_layer1.neurons
		hidden_layer2.neurons=5
		hidden_layer2.activation=self.props
		
		output_layer.inputDim=hidden_layer2.neurons
		output_layer.neurons=self.labels.shape[1]
		output_layer.activation=self.props
		
		batch_id=0
		begin=0
		end=0
		#training
		for i in xrange(self.max_itration):
			if end>=train_samples:
				#batch_id=begin%train_samples
				batch_id=0
			else:
				batch_id=batch_id+1
			begin=batch_id*self.batch
			end=(batch_id+1)*self.batch
			x=self.xAll[begin:end,:]
			y=self.labels[begin:end,:]
			print train_samples
			print end
			hidden_layer1.input=x
			hidden_layer1.samples=x.shape[0]
			hidden_layer1_output=hidden_layer1.forward()
			
			hidden_layer2.input=hidden_layer1_output
			hidden_layer2.samples=x.shape[0]
			hidden_layer2_output=hidden_layer2.forward()
			
			output_layer.input=hidden_layer2_output
			output_layer.samples=x.shape[0]
			output_layer_output=output_layer.forward()
			
			error=y-output_layer_output
			mse=self.props.errorFunc(error)
			print 'mse '+str(mse)+' iteration '+str(i+1)
			self.errorList.append(mse)
			if mse<=self.error_goal:
				self.iteration=i+1
				break;
			output_layer.error=error
			
			output_layer.layer_output=output_layer_output
			L2_error=output_layer.backword()
			
			hidden_layer2.layer_output=hidden_layer2_output
			hidden_layer2.error=L2_error
			L1_error=hidden_layer2.backword()
			
			hidden_layer1.layer_output=hidden_layer1_output
			hidden_layer1.error=L1_error
			hidden_layer1.backword()
		
		self.hidden_layer1=hidden_layer1
		self.hidden_layer2=hidden_layer2
		self.output_layer=output_layer
	
	def predict(self):
		x=normalize(self.test_set)
		self.hidden_layer1.input=x
		hidden1_output=self.hidden_layer1.forward()
		self.hidden_layer2.input=hidden1_output
		hidden2_output=self.hidden_layer2.forward()
		self.output_layer.input=hidden2_output
		output_output=self.output_layer.forward()
		return output_output
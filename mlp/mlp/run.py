import numpy as np
from layer import *
from props import *
from utils import *
'''
@Departured
'''

	
if __name__=="__main__":

	errorList=[]
	iteration=0
	error_goal=0.01
	
	props=props()
	data=loadDataSetgzip('../../data/mnist.pkl.gz')
	
	train_samples=10000
	xAll=data[0][0][0:train_samples]
	yAll=data[0][1][0:train_samples]
	#sigmoid needs the [-1,1] range, not the [0,1]
	xAll=normalize(xAll)
	
	train_set=xAll
	#mlp.labels=np.matrix(y).T
	labels=labelConvert(yAll,10)
	
	#initialize the network
	hidden_layer1=layer()
	hidden_layer2=layer()
	output_layer=layer()
	
	
	
	hidden_layer1.inputDim=xAll.shape[1]
	hidden_layer1.neurons=10
	hidden_layer1.activation=props
	
	hidden_layer2.inputDim=hidden_layer1.neurons
	hidden_layer2.neurons=5
	hidden_layer2.activation=props
	
	output_layer.inputDim=hidden_layer2.neurons
	output_layer.neurons=labels.shape[1]
	output_layer.activation=props
	
	max_itration=100
	batch=10
	#training
	for i in xrange(max_itration):
		if max_itration*batch>train_samples:
			batch_id=i%max_itration
		else:
			batch_id=i
		x=xAll[batch_id*batch:(batch_id+1)*batch,:]
		y=labels[batch_id*batch:(batch_id+1)*batch,:]
		
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
		mse=props.errorFunc(error)
		print 'mse '+str(mse)
		errorList.append(mse)
		if mse<=error_goal:
			iteration=i+1
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
	
	
	
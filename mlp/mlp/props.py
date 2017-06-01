import gzip
import numpy as np
import cPickle
'''
created on June 1 2017
@author: Zack
'''
class props(object):
	
	#def drawClassScatter(self,plt):
	
	#drawbackness this function max value only can be 1 when the label is 9,8 big value, all the prediction will be 1 because the least is 1 in my samples
	def activation(self,data):
		return 1.0/(1.0+np.exp(-data))

	def errorFunc(self,data):
		return np.sum(np.power(data,2))*0.5

	def derivative(self,data):
		return np.multiply(data,(1-data))
		
	
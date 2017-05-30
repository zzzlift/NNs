import numpy as np
from mlp import *

if __name__=="main":
	mlp=mlp()
	mlp.loadDataSet()
	mlp.train_set=mlp.normalize(mlp.train_set)
	mlp.test_set=mlp.normalize(mlp.test_set)
	
	mlp.bpTrain()
	
	mlp.BPClassifier()
	
	
	
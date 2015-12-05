import numpy as np
import os
import json



def loadMel(file="train_mel.db"):
	mels = []
	with open(file) as data_file:    
		temp = json.load(data_file)
		mels = np.array(temp)
	print len(mels)	

loadMel()
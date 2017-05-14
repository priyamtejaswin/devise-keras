''' This script tests modules in the project '''

import numpy as np
import h5py
import cPickle as pickle 

PATH_h5 = "processed_features/features.h5"
IMAGE_DIM = 4096
WORD_DIM = 50

def TEST_datagen():
	''' Testing the data_generator function in 
		extract_features_and_dump.py '''

	# Load Required data from disk
	F = h5py.File(PATH_h5, "r")
	image_fnames = F["data/fnames"][:] # list of list 
	image_fnames = list(map(lambda x: x[0], image_fnames))

	# Load module to Test
	from extract_features_and_dump import data_generator

	# Run module
	for x,y in data_generator(PATH_h5, batch_size=2):
		print x.shape, y.shape
		assert x.shape[0] == y.shape[0], "batch size should be same"
		assert x.shape[1] == IMAGE_DIM
		assert y.shape[1] == WORD_DIM

if __name__=="__main__":
	TEST_datagen()
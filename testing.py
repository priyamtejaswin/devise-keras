''' This script tests modules in the project '''

import numpy as np
import h5py
import cPickle as pickle 

PATH_h5 = "processed_features/features.h5"

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
		print 

if __name__=="__main__":
	TEST_datagen()
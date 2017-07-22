"""
USAGE: python extract_word_embeddings.py /path/to/raw_embeddings.txt /path/to/dump/data.h5

Script to extract word_embeddings from the raw file
and save to path.
"""

import sys
import os
import h5py
import numpy as np

from extract_features_and_dump import dump_wv_to_h5

embeddings_path, dump_path = sys.argv[1], sys.argv[2]
WORD_DIM = 300

if os.path.isfile(dump_path):
	res = raw_input("Found existing embedding dump file. Continue<y/n>?")
	if res=='n':
		sys.exit()

print "Loading glove vector data..."
F = h5py.File(dump_path, "w")
data = F.create_group("data")
dt   = h5py.special_dtype(vlen=str)
# extract and dump word vectors
print "Dumping word embeddings..."
_ = data.create_dataset("word_embeddings", (0, WORD_DIM), maxshape=(None, WORD_DIM))
_ = data.create_dataset("word_names", (0, 1), dtype=dt, maxshape=(None, 1))

glove_index = {}
with open(embeddings_path, 'r') as f:
	word_batch, vector_batch, _c = [], [], 1

	for line in f.readlines():
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    glove_index[word] = coefs
	    word_batch.append(word)
	    vector_batch.append(coefs)
	    
	    if _c%10000==0 and _c>0:
	    	dump_wv_to_h5(word_batch, vector_batch, F)
	    	print "Completed", _c
	    	word_batch, vector_batch = [], []

	    _c+=1

	dump_wv_to_h5(word_batch, vector_batch, F) # to catch the trailing vectors

F.close()
print 'Done. Found %s word vectors.' % len(glove_index)

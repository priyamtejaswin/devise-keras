import time
from flask import render_template, jsonify, request
from multiprocessing import Lock
from keras.models import load_model
from keras.preprocessing.text import text_to_word_sequence
import os, sys
import pickle
import h5py
from flask import Flask
app = Flask(__name__)

DUMMY_MODE=True
MODEL_LOC="snapshots/epoch_x.hdf5"

# VERY IMPORTANT VARIABLES
mutex = Lock()
MAX_SEQUENCE_LENGTH = 20
MODEL=None
DICT_word_index = None
if DUMMY_MODE==False:
	
	MODEL = load_model(MODEL_LOC)
	print MODEL.summary()
	
	assert os.path.isfile("DICT_word_index.TRAIN.pkl"), "Could not find DICT_word_index.TRAIN.pkl"	
	
	with open("DICT_word_index.TRAIN.pkl","r") as f:
		DICT_word_index = pickle.load(f)
	assert DICT_word_index is not None, "Could not load dictionary that maps word to index"

	im_outs = None 
	fnames = None
	with h5py.File("cache.h5") as F:
		im_outs = F["data/im_outs"][:]
		fnames  = F["data/fnames"][:]
	assert im_outs is not None, "Could not load im_outs from cache.h5"
	assert fnames is not None, "Could not load fnames from cache.h5"

# Query string -> word index list 
def query_string_to_word_indices(query_string):
	
	# string -> list of words 
	words = text_to_word_sequence(
			text = query_string,
			filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
			lower=True,
			split=" "
		)

	# check if words in dictionary
	all_words = DICT_word_index.keys()
	for word in words:
		assert word in all_words, "could not find word {} in all_words".format(word)

	# list of words -> list of indices
	words_index = []
	for word in words:
		words_index.append(DICT_word_index[word])

	# pad to 20 words
	if len(words_index) < MAX_SEQUENCE_LENGTH:
		padding = [0 for _ in range(MAX_SEQUENCE_LENGTH - len(words_index))]
		words_index += padding

	return np.array(words_index).reshape((1,MAX_SEQUENCE_LENGTH))

@app.route("/")
@app.route("/index")
def index():
	return render_template("index.html", title="Home")

def run_model(query_string):
	''' This fxn takes a query string
	runs it through the Keras model and returns result.'''

	# run forward pass
	# find diff 
	# get images having closest diff
	print "..waiting to acquire lock"
	result = None
	with mutex:
		print "lock acquired, running model..."
		if DUMMY_MODE:
			time.sleep(10)
			result = ["static/dog.jpg", "static/dog.jpg", "static/dog.jpg"]
		else:
			assert MODEL is not None, "not in dummy mode, but model did not load!"
			
			# forward pass 
			output = MODEL.predict([ np.zeros(1,4096) , query_string_to_word_indices(query_string) ])[:, WORD_DIM: ]
			output = output / np.linalg.norm(output, axis=1, keepdims=True)
			
			# compare with im_outs
			TOP_K = 50
			diff = im_outs - output 
			diff = np.linalg.norm(diff, axis=1)
			top_k_indices = np.argsort(diff)[:TOP_K].tolist()

			# populate "results" with fnames of top_k_indices
			result = []
			for k in top_k_indices:
				result.append(fnames[k])
			
		print '..over'
	
	if result is None:
		return 1,[]
	else:
		return 0,result

@app.route("/_process_query")
def process_query():

	query_string  	= request.args.get('query', type=str)
	rc, images 		= run_model(query_string) 

	result = {
		"rc":rc,
		"images": images
	}

	return jsonify(result)

if __name__ == '__main__':
	app.run(threaded=True)
import time
from flask import render_template, jsonify, request
from multiprocessing import Lock
from keras.models import load_model
from keras.preprocessing.text import text_to_word_sequence
import os, sys
import pickle
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

# Query string -> word index list 
def processing_query_string(query_string):
	
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
			# do something 

			result = []
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
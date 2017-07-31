import time
from flask import render_template, jsonify, request
from multiprocessing import Lock
from keras.models import load_model

from flask import Flask
app = Flask(__name__)

DUMMY_MODE=True
MODEL_LOC="snapshots/epoch_x.hdf5"

mutex = Lock()
MODEL=None
if DUMMY_MODE==False:
	MODEL = load_model(MODEL_LOC)

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
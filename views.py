import time
from flask import render_template, jsonify, request
from multiprocessing import Lock

from flask import Flask
app = Flask(__name__)

mutex = Lock()

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
		time.sleep(10)
		result = ["static/dog.jpg", "static/dog.jpg", "static/dog.jpg"]
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

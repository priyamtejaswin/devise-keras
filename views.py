from flask import render_template, jsonify, request

from flask import Flask
app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
	user = {
		"nickname": "akshay"
	}
	return render_template("index.html", title="Home", user=user)

@app.route("/_process_query")
def process_query():
	a = request.args.get('a', 0, type=int)
	b = request.args.get('b', 0, type=int)
	return jsonify(result=a+b)

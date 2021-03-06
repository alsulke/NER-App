from flask import Flask,jsonify
from flask_restful import Resource, Api
import wikipedia

app = Flask(__name__)

@app.route("/<string:name>")
def dummy_api(name:str) :
	context = wikipedia.summary(name)
	return jsonify(data=context)



if __name__ == '__main__':
	app.run()
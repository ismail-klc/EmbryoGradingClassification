import json
import os
from flask import Flask, jsonify, request
from flask_cors import CORS,cross_origin
from predict import predict

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/predict/", methods=['POST'])
def return_price():
    data = request.files['file'].read()
    pred = predict(data)
    pred_dict = {
        'result': pred,
    }
    return jsonify(pred_dict)

@app.route("/", methods=['GET'])
def default():
    return "<h1> Welcome to embryo grading model <h1>"


if __name__ == "__main__":
    app.run()

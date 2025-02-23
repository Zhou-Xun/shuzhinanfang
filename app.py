from crypt import methods
from pyexpat.errors import messages
from wsgiref.util import request_uri

from flask import Flask, request, jsonify

from algorithm.XGBoost import build_XGBoost, test_XGBoost

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello World!"

@app.route('/api/xgboostTrain', methods=['GET'])
def xgboostTrain():
    rmse = build_XGBoost()
    return jsonify(messages=f'xgboost model trained successfully! rmse: {rmse}')

@app.route('/api/xgboostTest', methods=['GET'])
def xgboostTest():
    rmse = test_XGBoost()
    return jsonify(messages=f'xgboost model tested successfully! rmse: {rmse}')

if __name__ == '__main__':
    app.run(port='8000', debug=True)
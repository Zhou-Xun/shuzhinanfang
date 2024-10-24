from crypt import methods
from pyexpat.errors import messages
from wsgiref.util import request_uri

from flask import Flask, request, jsonify

from algorithm.XGBoost import build_XGBoost

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello World!"

@app.route('/api/xgboostTrain', methods=['GET'])
def xgboostTrain():
    rmse = build_XGBoost()
    return jsonify(messages=f'xgboost model trained successfully! rmse: {rmse}')

if __name__ == '__main__':
    app.run(port='8050', debug=True)
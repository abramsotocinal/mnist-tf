import os
from flask import Flask, jsonify, Blueprint
from .include.predict import predict_random
from .include.train import train_model

# from . import app_blueprint

bp = Blueprint('main', __name__, url_prefix='/ml')

# @app.route('/randpredict', methods=['GET'])
@bp.route('/randpredict', methods=['GET'])
def randpredict():
  pred, actual = predict_random(os.path.join(bp.root_path,'models'))
  return jsonify(predicted=pred,actual=actual)

#@app.route('/train', methods=['GET'])
@bp.route('/train', methods=['GET'])
def train():
  return jsonify(training_summary=train_model(os.path.join(bp.root_path,'models')))

@bp.route('/')
def ml_root():
  return jsonify(message='hello')

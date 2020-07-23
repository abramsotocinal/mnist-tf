from flask import Flask, jsonify
from include.predict import predict_random
from include.train import train_model

app = Flask(__name__)

@app.route('/randpredict', methods=['GET'])
def randpredict():
  pred, actual = predict_random('models')
  return jsonify(predicted=pred,actual=actual)

@app.route('/train', methods=['GET'])
def train():
  return jsonify(training_summary=train_model())

# if __name__ == 'main':
app.run(port=5001)

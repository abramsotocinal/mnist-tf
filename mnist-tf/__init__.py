# flask application factory
import os
from flask import Flask, jsonify

def create_app(test_config=None):
  app = Flask(__name__, instance_relative_config=True)
  app.config.from_mapping(
    SECRET_KEY = 'dev'
  )
  if test_config is None:
    app.config.from_pyfile('config.py', silent=True)
  else:
    app.config.from_mapping(test_config)
  
  #ensure the instance folder exists
  try:
    os.makedirs(app.instance_path)
  except OSError as e:
    pass
    # raise('Cannot create dirs! {}'.format(e))
  @app.route('/test')
  def hello():
    return jsonify(message='App created')

  from .app import main
  app.register_blueprint(main.bp)
  return app
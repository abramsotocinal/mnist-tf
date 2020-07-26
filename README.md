# MNIST Fashion Dataset Classification with Keras As a Microservice

*Code taken from Tensorflow (tutorials)[https://www.tensorflow.org/tutorials/keras/classification]*

Classification implementation with Keras with an API endpoint in Flask

### Requirements

    Python 3.6+

### Instructions

Run the following commands

    $ pip install -r requirements.txt

    $ cd data && sh download_data.

Then, to run the service(need to be in project root)

    $ FLASK_APP=mnist-tf flask run



### Basic Usage

Run a random prediction

    GET /ml/randpredict

#### TODO:
- add more documentation
- add more tests
- containerize app, ready to be a microservice
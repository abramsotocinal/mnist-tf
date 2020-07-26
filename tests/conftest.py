import os
import importlib
import pytest
import sys

sys.path.append('..')
# didn't feel like renaming 'mnist-tf' to 'mnist_tf' (python module naming conventions)
mnist_tf = importlib.import_module('mnist-tf',None)


@pytest.fixture
def app():
    app = mnist_tf.create_app({
        'TESTING' : True
    })

    yield app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def runner(app):
    return app.test_cli_runner()
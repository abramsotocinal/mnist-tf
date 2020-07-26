import pytest

def test_routes(client,app):
  assert client.get('/ml/').status_code == 200

  assert client.get('/ml/randpredict').status_code == 200
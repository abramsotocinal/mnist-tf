from random import randint
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os

if __name__ == '__main__':
  from load_idx import Idx2Np
else:
  from include.load_idx import Idx2Np

# model_dir = '../models' if __name__ == '__main__' else 'models'

def predict_random(model_dir):
  # Labels
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

  # Load model
  model_name = 'MNIST_fashion'
  model = keras.models.load_model(os.path.join(model_dir,model_name))
  model.summary()

  probability_model = keras.Sequential([model, tf.keras.layers.Softmax()])
  idx = randint(0,10000)
  data = Idx2Np('test')
  data.unpack()
  input_image = data.array[idx]
  actual_label = data.label_array[idx]

  print('index: {}'.format(0))
  prediction = probability_model.predict(input_image.reshape(1,-1))
  print('predicted label: {}'.format(class_names[np.argmax(prediction)])) # the element with the highest probability will be considered the prediction
  print('actual label: {}'.format(class_names[actual_label]))
  
  return class_names[np.argmax(prediction)], class_names[actual_label]

def dummy():
  return 'shit'

if __name__ == '__main__':
  predict_random()
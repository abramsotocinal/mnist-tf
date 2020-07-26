import os
from tensorflow import keras
import tensorflow as tf


def train_model(model_dir):
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()
    # image pixels range from 0-255
    train_images = train_images / 255.0

    test_images = test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    # compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    # fit model
    model.fit(train_images, train_labels, epochs=5)

    # Evaluate
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)

    # save model
    model.save(os.path.join(model_dir,'MNIST_fashion'))

    return model.summary()

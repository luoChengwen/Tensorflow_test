import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.WARN)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D

import numpy as np
tf.enable_eager_execution()

(train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data()
a,b,c = np.shape(train_X)
train_X = train_X.reshape(a,b,c,1)
train_X = train_X / 255
a1, b1, c1 = np.shape(test_X)
test_X = test_X.reshape(a1,b1,c1,1)
test_X = test_X / 255

train_y = tf.one_hot(train_y, 10)
test_y = tf.one_hot(test_y, 10)

model = Sequential()
model.add(Conv2D(32,kernel_size = 5, activation='relu', input_shape= (28,28,1)))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(12, kernel_size =3, activation='relu'))
model.add(MaxPool2D(pool_size=2))
# model.add(Dropout(.1))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(.001), loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

train_X2= tf.convert_to_tensor(train_X, dtype=tf.float32)
model.fit(train_X2,train_y, batch_size=30, epochs=3)

# training loss, accuracy [0.0563 - categorical_accuracy: 0.9830]


test_X2 = tf.convert_to_tensor(test_X, dtype=tf.float32)
model.evaluate(test_X2, test_y)

# test loss, accuracy [0.04460997345575597, 0.9838]

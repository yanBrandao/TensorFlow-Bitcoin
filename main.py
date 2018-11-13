import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt
import pickle

print(tf.VERSION)
print(tf.keras.__version__)


model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))

# Create a sigmoid layer:
layers.Dense(64, activation='sigmoid')
# Or:
layers.Dense(64, activation=tf.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')

# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu'),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(4, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

              # Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

data = np.random.random((4, 4))
# 0 for Price / 1 for Amount / 2 for Hour / 3 for Date
labels = np.array([0, 1, 2, 3])

val_data = np.random.random((4, 4))
val_labels = np.array([0, 1, 2, 3])

# dataset = tf.data.Dataset.from_tensor_slices((tf.cast(data, tf.float32), tf.cast(labels, tf.float32)))
# dataset = dataset.batch(32).repeat()

# val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
# val_dataset = val_dataset.batch(32).repeat()

model.fit(data, labels, epochs=4, steps_per_epoch=30,
          validation_data=(val_data, val_labels),
          validation_steps=3)


data = np.random.random((4, 4))
labels = np.array([0, 1, 2, 3])

plt.plot(data)
plt.show()

model.evaluate(data, labels, steps=10)

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

result = model.predict(dataset, steps=32)
print(result.shape)
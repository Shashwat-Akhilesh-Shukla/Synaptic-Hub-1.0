# -*- coding: utf-8 -*-
"""CIFAR10 CNN CLASSIFICATION.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Q6Tg-QgZ_d2JpQF-UApxC0qrt5-wO2Vn

# IMPORT LIBRARIES
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, LeakyReLU

"""# LOAD AND PREPROCESS THE CIFAR10 DATASET"""

(train_images, train_labels), (test_images,test_labels) = cifar10.load_data()
train_images, test_images = train_images/255.0, test_images/255.0

"""# BUILD THE CONVOLUTION NEURAL NETWORK"""

model = Sequential()
model.add(Conv2D(32,(3,3), activation=None, input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3), activation=None, input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128,(3,3), activation=None, input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(128, activation=None))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(10, activation='softmax'))

"""# COMPILE THE MODEL"""

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

"""# TRAIN THE MODEL"""

history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

"""# EVALUATE ON THE TEST SET"""

test_loss, test_acc = model.evaluate(test_images,test_labels)
print(f"Test accuracy: {test_acc*100:.2f}%")
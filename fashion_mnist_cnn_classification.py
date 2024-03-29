# -*- coding: utf-8 -*-
"""Fashion MNIST CNN classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LGk_PbMoRRdZhiAmuu3__qQlay3aN5q4
"""

#import the libraries you require -- this is always the first step
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Check the shape of loaded images
print("Original shape of train_images:", train_images.shape)

# Reshape and normalize the images
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#build the CNN but make sure to not overcomplcate it by adding unnecessary layers
#complexity might increase or cause overfitting
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (28,28,1)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation = 'relu'))
model.add(layers.Dense(10,activation='softmax'))

#compile the model-- categorical classes and  thus the loss
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#training the model
history = model.fit(train_images, train_labels,epochs=10,batch_size=64,validation_split=0.2)

#now evaluate your model bro!
test_loss, test_acc = model.evaluate(test_images,test_labels)
print(f"Test accuracy:{test_acc*100:.2f}%")


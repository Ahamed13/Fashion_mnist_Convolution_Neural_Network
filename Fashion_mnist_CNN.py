import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfd
from PIL import Image
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels),(test_images, test_labels) = mnist.load_data()
categories = {"T-shirt/top": 0, "Trouser": 1,"Pullover":2, "Dress":3, "Coat":4, "Sandel": 5, "Shirt": 6, "Sneaker": 7, "Bag": 8, "Ankle_Boat" : 9}
list_categories = ['T-shirt/top','Trouser','Pullover','Dress', 'Coat','Sandel','Shirt', 'Sneaker', 'Bag', 'Ankle_Boat']
print(training_images.shape)
print(test_images.shape)
training_images = training_images.reshape(60000,28,28,1)
training_images = training_images/255.0
test_images = test_images.reshape(10000,28,28,1)
test_images = test_images/255.0
print(test_images.shape)
print(training_images.shape)
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(128,(3,3), activation='relu', input_shape=(28,28,1)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(training_images,training_labels, epochs = 15)
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
prediction = model.predict(test_images)
print(prediction[100])
max_val = np.argmax(prediction[100])
print(max_val)
print(list_categories[max_val])

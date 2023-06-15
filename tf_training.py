#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 09:29:56 2023

@author: arnd
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images1 = train_images
train_images2 = np.flip(train_images,2)
test_images1 = test_images
test_images2 = np.flip(test_images,2)
train_images = np.concatenate((train_images1, train_images2), axis=0)
test_images = np.concatenate((test_images1, test_images2), axis=0)

train_labels = np.append(train_labels, train_labels)
test_labels = np.append(test_labels, test_labels)
 
train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
# predictions[0]

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')



###################

def plot_img_and_value(testimg):
    plt.subplot(1,2,1)
    plt.imshow(testimg, cmap=plt.cm.binary)
    plt.subplot(1,2,2)
    plt.bar(class_names, np.reshape(probability_model.predict(testimg),(10,)))
    plt.xticks(rotation=90)


# image = Image.open('shoe_test.jpg')
image = Image.open('shirt_test.jpg')
new_image = image.resize((28, 28))
gray_image = new_image.convert('L')
img = np.asarray(gray_image)
img = img/255
img = (-1*img)+1

    
plot_img_and_value(img)    

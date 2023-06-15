#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:37:44 2023

@author: arnd
"""

import tensorflow as tf
from os import listdir
from os.path import isfile, join
import nibabel as nib
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean


##############################################################################


dat_folder = '/Users/arnd/Downloads/brain_train/'

allfilenames = []
filenames_t1w = []
filenames_t2w = []


allfilenames = [f for f in listdir(dat_folder) if isfile(join(dat_folder, f))]
for filename in allfilenames:
    if "_T1w" in filename:
        filenames_t1w += [filename]
    elif "_T2w" in filename:
        filenames_t2w += [filename]

t2_data = []
t2_data_out = []
for filename in filenames_t2w:
    nii_img  = nib.load(str(dat_folder + filename))
    t2_im = nii_img.get_fdata()
    t2_data_out = np.append(t2_data_out, resize(t2_im,(208,300,320)))
    t2_data = np.append(t2_data, resize(t2_im,(52,75,320)))
t2_data = np.reshape(t2_data,(52,75,-1))
t2_data_out = np.reshape(t2_data_out,(208,300,-1))

t1_data = []
for filename in filenames_t1w:
    nii_img  = nib.load(str(dat_folder + filename))
    t1_im = nii_img.get_fdata()
    t1_data = np.append(t1_data, resize(t1_im,(208,300,320)))
t1_data = np.reshape(t1_data,(208,300,-1))


##############################################################################


# Define the input layers
input1 = tf.keras.layers.Input(shape=(208, 300, 1))
input2 = tf.keras.layers.Input(shape=(52, 75, 1))
resized_input2 = tf.keras.layers.UpSampling2D(size=(4, 4))(input2)
input2 = resized_input2

# First convolutional layer for input1
conv1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(input1)
conv1_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(conv1_1)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1_2)

# First convolutional layer for input2
conv2_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(input2)
conv2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(conv2_1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv2_2)

# Concatenate the two convolutional layers
merged = tf.keras.layers.concatenate([pool1, pool2])

# Second convolutional layer
# conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(merged)
conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv3)

# Upsampling layer to increase the resolution of the image
upsample = tf.keras.layers.UpSampling2D(size=(2,2))(conv4)

# Final convolutional layer to generate the output image
output = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid')(upsample)

# Create the model
model = tf.keras.models.Model(inputs=[input1, input2], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


##############################################################################


# Generate some dummy training data
x1_train = np.random.rand(100, 208, 300)
#x2_train = np.random.rand(100, 52, 75)
x2_train = np.random.rand(100, 208, 300)
y_train = np.random.rand(100, 280, 300)

# Train the model
model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=32)
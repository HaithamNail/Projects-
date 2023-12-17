# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:42:08 2023

@author: cpu
"""

# Import necessary libraries
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, optimizers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

data_path = 'F:/spyder files/input_data'
train_path = 'F:/spyder files/train'
test_path = 'F:/spyder files/test'


# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(factor=0.2),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomZoom(height_factor=(0.2, 0.3), width_factor=(0.2, 0.3)),
    layers.RandomBrightness(factor=0.1),
    layers.RandomContrast(factor=0.1)
])


# Data loading and preprocessing
datagenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Assuming you want to split 20% for validation
)


batch_size = 32
train_generator = datagenerator.flow_from_directory(
    train_path,
    target_size=(300, 300),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

valid_generator = datagenerator.flow_from_directory(
    train_path,
    target_size=(300, 300),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


from tensorflow.keras.models import load_model

# Customizing the model based on InceptionV3
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

# Initializing InceptionV3 (pretrained) model with input image shape as (300, 300, 3)
base_model = InceptionV3(weights=None, include_top=False, input_shape=(300, 300, 3))

# Loading the weights after initializing the architecture
base_model.load_weights('F:\spyder files\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Setting the Training of all layers of InceptionV3 model to false
base_model.trainable = False

# Building the final model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.15),
    Dense(1024, activation='relu'),
    Dense(5, activation='softmax')  # 5 Output Neurons for 5 Classes
])

model.compiled_metrics == None
# Compiling the model after defining its architecture and loading weights
opt = optimizers.Adam(learning_rate=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])


# Separating Training and Testing Data
batch_size = 32
epochs = 10  # Adjust the number of epochs as needed


# Calculating variables for the model
steps_per_epoch = train_generator.n // batch_size
validation_steps = valid_generator.n // batch_size

# Displaying steps_per_epoch and validation_steps
print("steps_per_epoch :", steps_per_epoch)
print("validation_steps :", validation_steps)

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=validation_steps)
    
    
# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')


# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.show()

from tensorflow.keras.models import load_model
base_model.save('Modelv_2.h5') 

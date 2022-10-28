import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from tensorflow.python.estimator import keras
import keras

data_dir = r'G:\Weather\dataset'
categories = ['Cloud', 'Rain', 'Shine', 'Sunrise']
image_size = 100
training_data = []
for category in categories:
    path = os.path.join(data_dir, category)
    img_label = categories.index(category)
    for image in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, image))
            new_array = cv2.resize(img_array, (image_size, image_size))
            training_data.append([new_array, img_label])
        except Exception as e:
            pass

random.shuffle(training_data)
# plt.imshow(new_array)
# plt.show()

x = []
y = []

for features, labels in training_data:
    x.append(features)
    y.append(labels)

x = np.array(x).reshape(-1, image_size, image_size, 3)
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
y_train = keras.utils.to_categorical(y_train, 4)
y_test = keras.utils.to_categorical(y_test, 4)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(image_size, image_size, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=10)
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from keras.preprocessing import image

test_data_dir = r'G:\Skin Cancer\test'
test_categories = ['benign', 'malignant']
image_size = 100
testing_data = []
for test_category in test_categories:
    path_1 = os.path.join(test_data_dir, test_category)
    img1_label = test_categories.index(test_category)
    for test_image in os.listdir(path_1):
        try:
            img1_array = cv2.imread(os.path.join(path_1, test_image))
            new1_array = cv2.resize(img1_array, (image_size, image_size))
            testing_data.append([new1_array, img1_label])
        except Exception as e:
            pass

train_data_dir = r'G:\Skin Cancer\train'
train_categories = ['benign', 'malignant']
training_data = []
for train_category in train_categories:
    path_2 = os.path.join(train_data_dir, train_category)
    img2_label = train_categories.index(train_category)
    for train_image in os.listdir(path_2):
        try:
            img2_array = cv2.imread(os.path.join(path_2, train_image))
            new2_array = cv2.resize(img2_array, (image_size, image_size))
            training_data.append([new2_array, img2_label])
        except Exception as e:
            pass

random.shuffle(testing_data)
random.shuffle(training_data)
plt.imshow(new1_array)
plt.show()
plt.imshow(new2_array)
plt.show()

X_test = []
X_train = []
y_test = []
y_train = []

for test_features, test_label in testing_data:
    X_test.append(test_features)
    y_test.append(test_label)

for train_features, train_label in training_data:
    X_train.append(train_features)
    y_train.append(train_label)

X_test = np.array(X_test).reshape(-1, image_size, image_size, 3)
X_train = np.array(X_train).reshape(-1, image_size, image_size, 3)
y_test = np.array(y_test)
y_train = np.array(y_train)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(image_size, image_size, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=10)

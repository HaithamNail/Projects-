import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from sklearn.model_selection import train_test_split
from keras.preprocessing import image

data_dir = r'G:\kagglecatsanddogs_5340\PetImages'
categories = ['Dog', 'Cat']
image_size = 100
training_data = []

for category in categories:
    path = os.path.join(data_dir, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (image_size, image_size))
            training_data.append([new_array, class_num])
        except Exception as e:
            pass


random.shuffle(training_data)

plt.imshow(new_array, cmap='gray')
plt.show()

x = []
y = []

for features, labels in training_data:
    x.append(features)
    y.append(labels)

x = np.array(x).reshape(-1, image_size, image_size, 1)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 1)))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compiling model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

# Fitting model on training data
model.fit(x_train, y_train, epochs=1, batch_size=10)

test_image = image.load_img(r'F:\New folder\dogtest.jpg', target_size=(image_size, image_size))
test_image = test_image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)


if result[0][0] == 0:
    pred = 'Dog'
else:
    pred = 'Cat'

print(pred)

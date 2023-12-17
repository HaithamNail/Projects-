# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
import io
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'F:/spyder files/Modelv_2.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def predict(img_path, model):
    target_size = (300, 300) # Change this to your model's input size
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 # Normalize the image

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1) # Get the predicted class

    # Map class index to class label
    class_labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    predicted_label = class_labels[predicted_class[0]]

    return predicted_label

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('indexml.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        file = request.files['file']
        if file:
            # Save the file to a folder (if needed)
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
            file.save(file_path)

            # Make prediction
            prediction = predict(file_path, model)
            return jsonify({'class': prediction})
        else:
            return jsonify({'error': 'No file received'})
    return jsonify({'error': 'Invalid request'})

if __name__ == '__main__':
    app.run(debug=True)
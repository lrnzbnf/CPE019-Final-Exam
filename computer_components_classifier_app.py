# -*- coding: utf-8 -*-
"""computer_components_classifier_app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LMTkC7m3fUzFCsVFvwVFi5dqvLTuz62J
"""


import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('computer_components_classifier.h5')

# Define class names
class_names = list(train_generator.class_indices.keys())

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((img_height, img_width))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
st.title('Computer Components Classification App')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    image = preprocess_image(image)
    predictions = model.predict(image)
    score = tf.nn.softmax(predictions[0])
    st.write(f'This image is a {class_names[np.argmax(score)]} with a {100 * np.max(score):.2f} percent confidence.')

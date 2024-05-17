# streamlit_app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("Computer Component Classifier")

model = load_model('component_classifier.h5')

# Define class names directly here
class_names = ['case','cpu', 'gpu', 'hdd', 'keyboard', 'monitor', 'motherboard', 'mouse', 'ram']

def predict(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction = predict(img)
    max_index = np.argmax(prediction)
    if max_index < len(class_names):
        st.write(f"This is a {class_names[max_index]}")
    else:
        st.write("Sorry, could not classify the image into any known class.")

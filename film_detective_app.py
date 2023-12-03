import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


def resize_and_crop_image(image, size):
    target_ratio = size[0] / size[1]
    image_ratio = image.width / image.height

    if image_ratio > target_ratio:
        new_width = int(target_ratio * image.height)
        left = (image.width - new_width) / 2
        top = 0
        right = left + new_width
        bottom = image.height
    else:
        new_height = int(image.width / target_ratio)
        left = 0
        top = (image.height - new_height) / 2
        right = image.width
        bottom = top + new_height

    image = image.crop((left, top, right, bottom))
    image = image.resize(size, Image.LANCZOS)
    
    return image

def preprocess_image(image):
    processed_image = resize_and_crop_image(image, (600,600))
    processed_image = np.array(processed_image)
    processed_image = processed_image / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image
    
model = load_model('model_checkpoints/checkpoint_20-0.81.h5')


st.title('Film Detective')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Classify Image'):
        processed_image = preprocess_image(image)
    
        predictions = model.predict(processed_image)
    
        predicted_class = 'Film' if predictions[0][0] > 0.5 else 'Digital'
    
        st.write(f'Predicted Class: {predicted_class}')
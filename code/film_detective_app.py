import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
import config
from image_preprocessing import crop_resize

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    pooled_grads = tf.reduce_mean(grads, axis =(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def preprocess_image(image):
    processed_image = crop_resize(image, (700,700))
    processed_image = np.array(processed_image)
    processed_image = processed_image / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image
    
model = load_model('../model_checkpoints/checkpoint_20-0.81.h5')
last_conv_layer_name = config.last_conv_layer_name

st.title('Film Detective')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Classify Image'):
        processed_image = preprocess_image(image)
    
        predictions = model.predict(processed_image)
    
        predicted_class = 'Film' if predictions[0][0] > 0.5 else 'Digital'
    
        st.write(f'Predicted Class: {predicted_class}')
        
        heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer_name)
        
        st.write('Grad-CAM Heatmap:')
        st.image(heatmap, use_column_width=True)
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(current_dir, 'scripts')
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)
from image_preprocessing import crop_resize


def preprocess_image(image):
    processed_image = crop_resize(image, (700,700))
    processed_image = np.array(processed_image)
    processed_image = processed_image / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

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

def overlay_gradcam_heatmap(heatmap, uploaded_image):
    heatmap_expanded = np.expand_dims(heatmap, axis=-1)
    heatmap_resized = tf.image.resize(heatmap_expanded, (uploaded_image.size[1], uploaded_image.size[0]))
    heatmap_resized = np.array(heatmap_resized).squeeze()

    heatmap_min, heatmap_max = heatmap_resized.min(), heatmap_resized.max()
    if heatmap_max - heatmap_min > 0:
        heatmap_resized = (heatmap_resized - heatmap_min) / (heatmap_max - heatmap_min)
    heatmap_normalized = np.uint8(255 * heatmap_resized)

    heatmap_colored = cm.plasma(heatmap_normalized)[..., :3]
    heatmap_colored = Image.fromarray((heatmap_colored * 255).astype(np.uint8))

    combined_image = Image.blend(uploaded_image.convert('RGB'), heatmap_colored, alpha=0.5)
    return combined_image

model_path = os.path.join(os.path.dirname(__file__), 'models', 'checkpoint_20-0.81.h5')
model = load_model(model_path)

last_conv_layer_name = 'conv2d_4'

st.title('Film Detective 🔍')

st.write('What is the origin of your image? Find out if your photo was shot with a film or digital camera through the power of deep learning!')

uploaded_file = st.file_uploader('Upload a digital or film photo...', type=['jpg', 'jpeg'])

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Investigate Image'):
        processed_image = preprocess_image(uploaded_image)
    
        predictions = model.predict(processed_image)
    
        predicted_class = 'Film' if predictions[0][0] > 0.5 else 'Digital'
        
        st.subheader('Result:')
        
        if predicted_class == 'Film':
            st.success(f'Your image was likely shot on film!')
        else:
            st.success(f'Your image was most likely shot on a digital camera or created with digital techniques.')
        
        heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer_name)
        combined_image = overlay_gradcam_heatmap(heatmap, uploaded_image) 
        
        st.subheader('Grad-CAM Heatmap')
        st.write('This heatmap highlights which parts of your image were most important to the deep learning model in its decision:')
        st.image(combined_image, use_column_width=True)


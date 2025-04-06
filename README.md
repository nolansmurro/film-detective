# Film vs. Digital Photo Classifier

## Problem Statement
The project aims to develop a Convolutional Neural Network (CNN) model that can accurately classify images as either film or digital. By leveraging deep learning techniques, this model seeks to assist photographers, archivists, digital image analysts, and image hosting services in efficiently categorizing photographic content and understanding its origins.

## Executive Summary
This project presents a CNN model capable of distinguishing between film and digital photography. Utilizing a dataset of images processed and augmented through various Python scripts, the model undergoes training to learn the distinctive features of each category. A Streamlit web application integrates the model, offering users an interactive platform to upload images, receive predictions, and view Grad-CAM heatmaps highlighting influential image regions for model interpretability.

### Key Components
- Image preprocessing and augmentation scripts for dataset preparation.
- CNN model architecture with multiple convolutional, max-pooling, dropout, and dense layers.
- Grad-CAM heatmap generation for interpretability.
- Streamlit web application for user interaction and visualization.

<!-- ## File Directory / Table of Contents
```
- README.md
- code/
  - image_preprocessing.py
  - cnn_model_script.py
  - streamlit_app.py
- datasets/
  - film/
  - digital/
- models/
  - checkpoint_20-0.81.h5
- requirements.txt
``` -->

<!-- ## Installation and Setup
1. Clone the repository.
2. Install the required libraries: `pip install -r requirements.txt`.
3. Run the Streamlit app: `streamlit run streamlit_app.py`.

## Data Dictionary
| Feature       | Description                              |
|---------------|------------------------------------------|
| Image         | JPEG image files                  |
| Label         | Categorized as 'Film' or 'Digital'       | -->

## Model Training and Evaluation
The CNN model is trained on the preprocessed images, with an architecture consisting of convolutional layers for feature extraction and dense layers for classification. The model's performance is evaluated based on accuracy, and Grad-CAM heatmaps are generated for interpretability.

## Streamlit Web Application
The Streamlit app provides an interface for users to upload images and receive predictions. It visualizes Grad-CAM heatmaps, offering insights into the model's decision-making process.

### Using the App
- Upload an image using the provided uploader.
- Click on 'Investigate Image' to get the prediction.
- View the classification result and the corresponding heatmap.

## Conclusions and Recommendations
The project demonstrates the potential of CNNs in image classification tasks, particularly in distinguishing between film and digital photos. Future improvements could include expanding the dataset, refining the model architecture, and enhancing the Streamlit app's functionality.

## Sources
<!-- - Image dataset: [Specify dataset source if applicable] -->
- TensorFlow and Keras for model building.
- PIL, NumPy, and Matplotlib for image processing and visualization.

---

This README provides a comprehensive overview of your project, ensuring clarity and ease of understanding for users and contributors. Feel free to modify or expand any section to better fit your project's specifics.
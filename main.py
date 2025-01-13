import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import cv2
import os

# Load the precomputed features and filenames
try:
    feature_list = np.array(pickle.load(open('/your/file/path/embeddings.pkl', 'rb')))
    filenames = pickle.load(open('/Users/bishal/your/file/path/file_names.pkl', 'rb'))
except Exception as e:
    st.error(f"Failed to load embeddings or filenames: {e}")
    st.stop()

# Define the ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Streamlit UI
st.title("Image Recommendation System")
st.write("Provide an image URL or upload an image, and we will show similar images based on the feature embeddings.")

# Input options
image_url = st.text_input("Enter the URL of an image:")
uploaded_file = st.file_uploader("Or upload an image...", type=["jpg", "jpeg", "png"])

# Function to load and validate an image from a URL
def process_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Validate Content-Type header
        content_type = response.headers.get("Content-Type")
        if "image" not in content_type:
            st.error("The URL does not point to a valid image.")
            return None

        # Load and return the image
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the image: {e}")
        return None

    except UnidentifiedImageError:
        st.error("The provided URL does not point to a valid image file.")
        return None

# Check for input (URL or file upload)
if image_url or uploaded_file:
    if image_url:
        img = process_image_from_url(image_url)
    elif uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
    
    if img:
        st.image(img, caption='Input Image', use_column_width=True)

        # Preprocess the input image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        expand_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expand_img_array)

        # Extract features
        result = model.predict(preprocessed_img).flatten()
        norm_result = result / norm(result)

        # Find nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([norm_result])

        # Display recommended images
        st.write("### Recommended Images:")
        cols = st.columns(5)

        base_path = "/your/image/folder/path/"  # Update this to your images folder

        for i, file_idx in enumerate(indices[0]):
            file_path = os.path.join(base_path, filenames[file_idx])  # Construct the full path

            # Verify file path exists
            if not os.path.exists(file_path):
                st.error(f"File does not exist: {file_path}")
                continue

            # Load and display the image
            temp_img = cv2.imread(file_path)
            if temp_img is None:
                st.error(f"Could not read image from path: {file_path}. Ensure the file is a valid image.")
                continue

            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(temp_img, (200, 200))

            # Display each recommended image in its column
            with cols[i]:
                st.image(resized_img, caption=f"Image {i+1}", use_column_width=True)

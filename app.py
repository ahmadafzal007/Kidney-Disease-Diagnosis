import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f0f0f5;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        color: #2c3e50;
        margin-top: -50px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #7f8c8d;
        margin-bottom: 50px;
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #bdc3c7;
        margin-top: 50px;
    }
    .uploaded-image {
        border: 2px solid #2c3e50;
        border-radius: 10px;
        margin: 20px 0;
    }
    .prediction-box {
        border: 2px solid #27ae60;
        background-color: #1f1e1c;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the models
vgg_model = tf.keras.models.load_model('Models/VGG_model.h5')
resnet_model = tf.keras.models.load_model('Models/ResNet_model.h5')
mobilenet_model = tf.keras.models.load_model('Models/MobileNet_model.h5')

# Define class names
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Function to preprocess the image
def preprocess_image(image):
    image = np.array(image)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions with all three models
def make_predictions(image):
    vgg_pred = vgg_model.predict(image)
    resnet_pred = resnet_model.predict(image)
    mobilenet_pred = mobilenet_model.predict(image)
    
    vgg_conf = np.max(vgg_pred) * 100
    resnet_conf = np.max(resnet_pred) * 100
    mobilenet_conf = np.max(mobilenet_pred) * 100
    
    vgg_label = class_names[np.argmax(vgg_pred)]
    resnet_label = class_names[np.argmax(resnet_pred)]
    mobilenet_label = class_names[np.argmax(mobilenet_pred)]
    
    return {
        'VGG': {'label': vgg_label, 'confidence': vgg_conf},
        'ResNet': {'label': resnet_label, 'confidence': resnet_conf},
        'MobileNet': {'label': mobilenet_label, 'confidence': mobilenet_conf},
    }

# Function to find the most probable result
def most_probable_result(predictions):
    labels = [predictions['VGG']['label'], predictions['ResNet']['label'], predictions['MobileNet']['label']]
    most_common = max(set(labels), key=labels.count)
    return most_common

# Streamlit app layout
st.markdown('<h1 class="title">Kidney Disease Detection</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">Upload an image of a kidney scan to get predictions from three different models.</h3>', unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make predictions
    predictions = make_predictions(processed_image)
    
    # Display predictions and confidence scores
    st.markdown("### Predictions:")
    for model_name, result in predictions.items():
        st.markdown(f"""
        <div class="prediction-box">
            <strong>{model_name}</strong>: {result['label']} with confidence <strong>{result['confidence']:.0f}%</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Determine the most probable result
    probable_result = most_probable_result(predictions)
    st.markdown(f"### Most Probable Result: <strong>{probable_result}</strong>", unsafe_allow_html=True)

st.markdown('<div class="footer">© 2024 Kidney Disease Detection App</div>', unsafe_allow_html=True)

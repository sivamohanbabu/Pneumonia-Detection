import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load your pre-trained model (ensure the correct file path)
model = load_model('models/pneumonia_model.keras')  # Or 'pneumonia_model.h5'

# Recompile the model (important if it wasn't compiled when saved)
model.compile(optimizer='adam',  # Or the same optimizer used during training
              loss='binary_crossentropy',  # Or the loss function used during training
              metrics=['accuracy'])  # Or the metrics used during training

# Function to preprocess and make predictions
def predict_image(img_path):
    # Open the image file
    img = Image.open(img_path)
    
    # Resize the image to 150x150 (the input size expected by your model)
    img = img.resize((150, 150))
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Ensure the image has 3 channels (RGB)
    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1)  # Duplicate the single channel to 3 channels
    
    # Normalize the image (convert values to between 0 and 1)
    img_array = img_array / 255.0
    
    # Add batch dimension (model expects batch dimension as the first dimension)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make the prediction
    prediction = model.predict(img_array)
    
    # Return the prediction and the prediction probability
    return prediction

# Custom CSS for Streamlit UI
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
            color: #333;
        }
        .title {
            font-size: 2rem;
            font-weight: 700;
            color: #4CAF50;
        }
        .sidebar {
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content {
            color: #000;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stFileUploader>label {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit sidebar
with st.sidebar:
    st.header("About this App")
    st.write("""
        This is a Pneumonia Detection app. Upload a chest X-ray image, and the model will predict whether the image shows signs of pneumonia or is normal. 
        The model uses deep learning and is trained on a dataset of chest X-rays.
    """)
    st.write("### Instructions:")
    st.write("1. Upload an X-ray image.")
    st.write("2. Get the prediction along with the confidence percentage.")

# Streamlit main UI
st.title("Pneumonia Detection")
st.write("Upload a chest X-ray image to predict if it shows pneumonia or not.")

# Upload image
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image and result side by side
    col1, col2 = st.columns([1, 2])

    with col1:
        # Display uploaded image in a smaller size
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=200)

    with col2:
        # Make prediction
        prediction = predict_image(uploaded_file)

        # Display the result with additional parameters
        probability = prediction[0][0]  # Probability of having pneumonia (assuming binary classification)
        
        # Display prediction
        if probability > 0.5:
            st.write("### Prediction: Pneumonia")
            st.write(f"**Confidence**: {probability * 100:.2f}%")
            st.markdown('<p style="color: red;">This may indicate a case of pneumonia, please consult a doctor for further tests.</p>', unsafe_allow_html=True)
        else:
            st.write("### Prediction: Normal")
            st.write(f"**Confidence**: {(1 - probability) * 100:.2f}%")
            st.markdown('<p style="color: green;">The X-ray appears to be normal. No signs of pneumonia detected.</p>', unsafe_allow_html=True)
        
        # Display raw prediction (for debugging or deeper analysis)
        st.write(f"**Raw Prediction Value**: {probability}")

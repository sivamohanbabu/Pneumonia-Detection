import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from difflib import get_close_matches

# Load your pre-trained model (ensure the correct file path)
model = load_model('models/pneumonia_model.keras')  # Replace with your model's actual path

# Recompile the model (important if it wasn't compiled when saved)
model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Function to preprocess and make predictions
def predict_image(img_path):
    img = Image.open(img_path)
    img = img.resize((150, 150))
    img_array = np.array(img)

    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

# Load 1,000 FAQs from a text file or define a dictionary
import json
with open("faqs.json", "r") as f:
    faq_data = json.load(f)

# Chatbot function to match user questions with FAQs
def faq_chatbot(user_input):
    # Find the closest match from the FAQ dataset
    closest_match = get_close_matches(user_input, faq_data.keys(), n=1, cutoff=0.5)
    if closest_match:
        return faq_data[closest_match[0]]
    else:
        return "Sorry, I couldn't find an answer to your question. Please consult a medical professional."

# Add Tailwind CSS for styling with updated class names
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #e0f7fa;  /* Soft light blue background */
        }
        .header-title {
            font-size: 2.5rem;
            color: #00796b;  /* Deep teal */
            font-weight: bold;
        }
        .sidebar-title {
            color: #0288d1;  /* Blue for sidebar titles */
        }
        #stSidebarContent{
            color:"red"
            background:"#229fd8"
            }
        .stButton>button, .stFileUploader>label {
            background-color: #039be5 !important;  /* Bright blue */
            color: white !important;
            font-weight: bold;
            border-radius: 10px;
            padding: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stTextInput input {
            border-radius: 8px;
            padding: 12px;
            font-size: 1.1rem;
            border: 2px solid #00796b;  /* Teal border for input fields */
        }
        .stTextInput input:focus {
            border-color: #0288d1;  /* Blue focus border */
            box-shadow: 0 0 5px #0288d1;
        }
        .stTextArea textarea {
            border-radius: 8px;
            padding: 12px;
            font-size: 1.1rem;
            border: 2px solid #00796b;
        }
        .response-message {
            font-weight: bold;
            font-size: 1.1rem;
            color: #d32f2f;  /* Red for warning or advice messages */
        }
        .normal-response {
            color: #388e3c;  /* Green for normal, healthy results */
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit sidebar with navigation options
with st.sidebar:
    st.image("./Asset/logo.png")
    st.title("Your Health, Our Priority")
    st.write("""This app predicts pneumonia based on X-ray images and answers questions about pneumonia.""")
        # Sidebar navigation options
    option = st.radio("Choose a feature:", ("Pneumonia Detection", "Pneumonia FAQ Chatbot"))

    st.write("### Instructions:")
    st.write("1. Upload an X-ray image.")
    st.write("2. Get a prediction with a confidence percentage.")
    st.write("3. Ask any questions about pneumonia.")


# Conditional display based on the sidebar selection
if option == "Pneumonia Detection":
    # Pneumonia Detection Section
    st.markdown('<h1 class="header-title">🩺 Pneumonia Detection</h1>', unsafe_allow_html=True)
    st.write("Upload a chest X-ray image to predict pneumonia.")

    # Image upload and prediction
    uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])

        with col1:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", width=200)

        with col2:
            prediction = predict_image(uploaded_file)
            probability = prediction[0][0]

            if probability > 0.5:
                st.write("### Prediction: Pneumonia")
                st.write(f"**Confidence**: {probability * 100:.2f}%")
                st.markdown('<p class="response-message">Consult a doctor for further evaluation.</p>', unsafe_allow_html=True)
            else:
                st.write("### Prediction: Normal")
                st.write(f"**Confidence**: {(1 - probability) * 100:.2f}%")
                st.markdown('<p class="normal-response">No signs of pneumonia detected.</p>', unsafe_allow_html=True)

elif option == "Pneumonia FAQ Chatbot":
    # Chatbot Section (after detection)
    st.markdown('<h2 class="header-title">🤖 Pneumonia FAQ Chatbot</h2>', unsafe_allow_html=True)
    user_question = st.text_input("Ask a question about pneumonia:")

    if user_question:
        response = faq_chatbot(user_question)
        st.write(f"**Bot Response:** {response}")

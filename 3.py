import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

# Directory and CSV setup
SAVE_DIR = "user_data"
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_FILE = os.path.join(SAVE_DIR, "patient_data.csv")

# Initialize CSV file if not exists
if not os.path.exists(CSV_FILE):
    columns = ["Name", "Age", "Tender/swollen anterior cervical lymph nodes", "Patient ID", "Phone", "Fever", "Cough", "Prediction", "Doctor's Decision", "Image Path"]
    pd.DataFrame(columns=columns).to_csv(CSV_FILE, index=False)

# Load the pre-trained model
device = torch.device("cpu")  # CPU-only

# Define the model architecture
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 1),
    nn.Sigmoid()
)

# Load model weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))  # Make sure to load the model on CPU
model = model.to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to predict image class
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        prediction = output.item()
    return prediction

# Streamlit UI
st.title("Antibiotic Requirement Evaluation in Sore Throat (A.R.E.S.T)")
st.markdown("Fill in patient details, upload an image, and save results. The model will predict whether antibiotics are required based on the image.")

# Use a two-column layout for input fields
col1, col2 = st.columns(2)

# Input fields for patient information
with col1:
    name = st.text_input("Name", key="name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1, key="age")
    lymph_nodes = st.selectbox("Tender/swollen anterior cervical lymph nodes", ["Yes", "No"], key="lymph_nodes")
    fever = st.selectbox("Fever", ["Yes", "No"], key="fever")

with col2:
    patient_id = st.text_input("Patient ID", key="patient_id")
    phone = st.text_input("Phone Number", key="phone")
    cough = st.selectbox("Cough", ["Yes", "No"], key="cough")

# File uploader for image
file_uploader = st.file_uploader("Choose an image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

# Display uploaded image
if file_uploader:
    image = Image.open(file_uploader)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict image class
    prediction = predict_image(image)
    if prediction > 0.5:
        prediction_class = "Antibiotic required"
        st.markdown(f"### **{prediction_class}**", unsafe_allow_html=True)
    else:
        prediction_class = "Antibiotic not required"
        st.markdown(f"### **{prediction_class}**", unsafe_allow_html=True)

    # Add doctor's decision
    doctor_decision = st.selectbox("Doctor's Decision", 
                                  ["Antibiotic required", "Not required", "Rejected", "Throat swab first", "Wait and watch"])

# Submit button with feedback
submit_button = st.button("Submit", help="Click here to save the data and results")

if submit_button:
    if name and age and lymph_nodes and patient_id and phone and fever and cough and file_uploader and doctor_decision:
        # Save image locally
        image_path = os.path.join(SAVE_DIR, f"{patient_id}_{file_uploader.name}")
        image.save(image_path)
        
        # Save patient details to CSV
        new_entry = pd.DataFrame([{
            "Name": name,
            "Age": age,
            "Tender/swollen anterior cervical lymph nodes": lymph_nodes,
            "Patient ID": patient_id,
            "Phone": phone,
            "Fever": fever,
            "Cough": cough,
            "Prediction": prediction_class,
            "Doctor's Decision": doctor_decision,
            "Image Path": image_path
        }])
        
        # Append to CSV using pd.concat()
        existing_data = pd.read_csv(CSV_FILE)
        updated_data = pd.concat([existing_data, new_entry], ignore_index=True)
        updated_data.to_csv(CSV_FILE, index=False)
        
        st.success("Recorded successfully!")
    else:
        st.error("Please fill all the required fields and upload an image before submitting.")

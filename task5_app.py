import streamlit as st
import cv2, numpy as np, joblib
from feature_extraction import extract_features
from labels import label_map

"""
README:
1.	Install dependencies (pip install streamlit opencv-python scikit-learn joblib).
2.	Navigate to the project folder.
3.	Run "streamlit run task5_app.py".
4.	Upload an image
5.	The app will display the image and show the predicted class.
"""

# Load model
model = joblib.load("svm_linear_final.pkl")

st.title("Plant Classification App")
st.write("Upload an image of a plant. Then the app will predict its class.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Show image
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    # Extract features & predict
    features = extract_features(img) 
    if features is not None:
        pred = model.predict([features])[0]
        st.success(f"Prediction: {label_map[pred]}")
    else:
        st.error("Could not extract features from this image.")

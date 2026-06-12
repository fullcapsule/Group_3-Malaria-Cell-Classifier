import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Malaria Diagnostic Lab",
    page_icon="🩸",
    layout="wide"  # Opens up the screen for a side-by-side dashboard look
)

# Load the model once and cache it
@st.cache_resource
def load_malaria_model():
    model_path = r"C:\Users\Bright\Desktop\ds_today\project\models\malaria_final_production.keras"
    return tf.keras.models.load_model(model_path)

try:
    model = load_malaria_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# ==========================================
# 2. SIDEBAR INFO PANEL
# ==========================================
with st.sidebar:
    st.header("🔬 Lab Specifications")
    st.write("---")
    st.markdown("**Core Brain:** MobileNetV2")
    st.markdown("**Training Type:** Fine-Tuned (Top 20 Layers)")
    st.markdown("**Verified Accuracy:** `96.0%` on Test Set")
    st.write("---")
    st.info("💡 **Tip:** Use thin, clear blood smear images for the best performance.")

# ==========================================
# 3. MAIN HEADER PAGE
# ==========================================
st.title("🩸 Malaria Cell Image Classifier")
st.write("Welcome to the automated diagnosis system. Drop an image below to run the AI system.")
st.write("---")

# File uploader centered on screen
uploaded_file = st.file_uploader(
    "Drag and drop a cell image here...", 
    type=["png", "jpg", "jpeg"]
)

# ==========================================
# 4. SIDE-BY-SIDE DASHBOARD LAYOUT
# ==========================================
if uploaded_file is not None:
    st.write("") # Add space
    
    # Create two equal columns on the screen
    col1, col2 = st.columns(2)
    
    # Left Column: Show the Image
    with col1:
        st.subheader("🖼️ Uploaded Cell")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption=f"File: {uploaded_file.name}")
        
    # Right Column: Show the AI Prediction
    with col2:
        st.subheader("📊 AI Analysis")
        
        with st.spinner("Scanning cell pixels..."):
            # Process image matching your 224x224 input
            img_resized = image.resize((224, 224))
            img_array = tf.keras.utils.img_to_array(img_resized)
            img_batch = np.expand_dims(img_array, axis=0)
            img_processed = preprocess_input(img_batch)
            
            # Predict
            probability = model.predict(img_processed)
            
            # Display clean, color-coded response cards
                    # Display clean, color-coded response cards
            if probability > 0.5:
               
                confidence = float(probability[0][0] * 100)
                
                st.success("### ✨ Result: UNINFECTED (Healthy Cell)")
                st.metric(label="AI Confidence Score", value=f"{confidence:.1f}%")
                st.write("The network found no traces of malaria ring-stages in this sample.")
                st.warning("⚠️ Clinical Notice: Please verify this diagnosis with a professional lab practitioner.")
            else:

                confidence = float((1 - probability[0][0]) * 100)
                
                st.error("### 🚨 Result: PARASITIZED (Malaria Parasite Detected)")
                st.metric(label="AI Confidence Score", value=f"{confidence:.1f}%")
                st.warning("⚠️ Critical: High probability of parasite presence. Double check this slice.")
                st.warning("⚠️ Clinical Notice: Please verify this diagnosis with a professional lab practitioner.")
            
        st.write("---")
        
        # 5. GENERATE COMPACT TEXT REPORT BUTTON
        report_text = f"MALARIA DIAGNOSTIC REPORT\n" \
                      f"--------------------------\n" \
                      f"Image Name: {uploaded_file.name}\n" \
                      f"Classification: {'UNINFECTED' if probability > 0.5 else 'PARASITIZED'}\n" \
                      f"Model Confidence: {confidence:.1f}%\n"
                      
        st.download_button(
            label="📥 Export Report Summary",
            data=report_text,
            file_name=f"malaria_report_{uploaded_file.name}.txt",
            mime="text/plain",
            use_container_width=True # Stretches the button to fit the column beautifully
        )

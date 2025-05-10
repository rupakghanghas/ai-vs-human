import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import cv2
import time

# Load model and define constants
model = load_model('MobileNetV2_finetuned_model(0.95 loss 0.11).keras')
IMAGE_SIZE = (224, 224)

# Background setup
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: url("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg");
            background-size: cover;
            backdrop-filter: blur(10px);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Preprocess input image
def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize(IMAGE_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image / 255.0

# Grad-CAM Heatmap Generation
def generate_gradcam_heatmap(model, img_array, last_conv_layer_name="out_relu", pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# Overlay heatmap on image
def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

# Classification and Grad-CAM handler
def classify_with_gradcam(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    class_label = int(np.round(prediction[0][0]))
    confidence = prediction[0][0]
    heatmap = generate_gradcam_heatmap(model, processed)
    cam_image = overlay_heatmap_on_image(image, heatmap)
    return class_label, confidence, cam_image

# Set up UI
set_background()
st.markdown("<h1 style='text-align: center; font-size: 48px;'>AI vs Real Art Image Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px;'>Upload image(s) and see if they are AI-generated or real artwork with Grad-CAM explanation.</p>", unsafe_allow_html=True)

# Model Information
with st.expander("📊 Model Details"):
    st.write("""
    - Model: Fine-tuned MobileNetV2  
    - Training Accuracy: 94.5%  
    - Validation Accuracy: 95.88%  
    - Dataset: 27,266 total images  
    - Grad-CAM is used to visualize focus areas of the model during prediction.
    """)

# Image Upload Section
uploaded_files = st.file_uploader("Upload your image(s) here:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if 'page' not in st.session_state:
        st.session_state.page = 0

    current_index = st.session_state.page
    total_images = len(uploaded_files)
    image = Image.open(uploaded_files[current_index])

    # Simulated Loading
    with st.spinner("Processing image..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

    st.success("Image uploaded successfully!")

    # Classify and Visualize
    label, confidence, cam_image = classify_with_gradcam(image)

    # Display Inputs and Outputs
    st.markdown("### Result")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption=f"Uploaded Image - {uploaded_files[current_index].name}", use_container_width=True)

    with col2:
        result_text = "AI Art" if label == 0 else "Real Art"
        color = "red" if label == 0 else "green"
        st.markdown(f"<h2 style='color:{color}; text-align:center;'>{result_text}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>Confidence: {confidence:.4f}</p>", unsafe_allow_html=True)
        st.image(cam_image, caption="Grad-CAM Visualization", use_container_width=True)

    # Navigation buttons
    nav1, _, nav2 = st.columns([1, 2, 1])
    if current_index > 0 and nav1.button("⬅ Previous"):
        st.session_state.page -= 1
    if current_index < total_images - 1 and nav2.button("Next ➡"):
        st.session_state.page += 1

    # Reset Button
    st.markdown(
        """
        <div style='text-align: center; margin-top: 20px;'>
            <form action="/" method="get">
                <button style="font-size: 18px; padding: 10px 20px;">Clear All and Start Over</button>
            </form>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    """
    <hr>
    <div style='text-align: center; font-size: 16px;'>
        © 2024 | Developed by 
        <a href='https://github.com/rupakghanghas' target='_blank'>
            <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Github-desktop-logo-symbol.svg/2048px-Github-desktop-logo-symbol.svg.png' height='20'>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

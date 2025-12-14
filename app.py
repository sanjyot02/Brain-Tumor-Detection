import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("best_model.keras")

# Class labels
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMAGE_SIZE = (150, 150)

# Preprocessing
def preprocess_image(image):
    image = image.convert("RGB")
    img = image.resize(IMAGE_SIZE)         
    img_array = np.array(img) / 255.0      
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Auto-detect last Conv2D layer
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")

# Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Overlay heatmap
def overlay_heatmap(heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    img = np.array(image)
    superimposed_img = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    return Image.fromarray(superimposed_img)

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor MRI Classifier")
st.write("A CNN-based application for automated brain tumor classification from MRI scans.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", width="stretch")

    # Prediction
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    confidence = np.max(prediction)
    predicted_class = categories[np.argmax(prediction)]

    # Show results
    st.subheader("Prediction Result:")

    if predicted_class in ["glioma", "meningioma", "pituitary"]:
        st.write("🛑 **Tumor Detected**")
        st.write(f"**Tumor Type:** {predicted_class.upper()}")
    else:
        st.write("✅ **No Tumor Detected**")

    st.write(f"**Confidence:** {confidence:.2f}")
    st.bar_chart(dict(zip(categories, prediction[0])))

    # Grad-CAM
    last_conv_layer = get_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(processed_img, model, last_conv_layer)
    gradcam_img = overlay_heatmap(heatmap, image)

    st.subheader("Grad-CAM Heatmap")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original MRI", width="stretch")
    with col2:
        st.image(gradcam_img, caption="Model Focus (Grad-CAM)", width="stretch")

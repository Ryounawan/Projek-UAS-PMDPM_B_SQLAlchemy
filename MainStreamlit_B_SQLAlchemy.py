import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Muat model
model = load_model('gugelnet.h5')
class_names = ['Matang', 'Mentah']

def classify_image(image):
    try:
        # Proses gambar
        input_image = image.resize((180, 180))  # Resize gambar
        input_image_array = tf.keras.utils.img_to_array(input_image) / 255.0
        input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

        # Prediksi
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])

        # Klasifikasi
        class_idx = np.argmax(result)
        confidence_scores = result.numpy()
        return class_names[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

def custom_progress_bar(confidence, color1, color2):
    percentage1 = confidence[0] * 100
    percentage2 = confidence[1] * 100
    progress_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px;">
        <div style="width: {percentage1:.2f}%; background: {color1}; color: white; text-align: center; height: 24px; float: left;">
            {percentage1:.2f}%
        </div>
        <div style="width: {percentage2:.2f}%; background: {color2}; color: white; text-align: center; height: 24px; float: left;">
            {percentage2:.2f}%
        </div>
    </div>
    """
    st.sidebar.markdown(progress_html, unsafe_allow_html=True)

st.title("Prediksi Kematangan Buah Naga - 10813")

uploaded_files = st.file_uploader("Unggah Gambar (Beberapa diperbolehkan)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if st.sidebar.button("Prediksi"):
    if uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            try:
                # Baca gambar langsung dari `uploaded_file`
                image = Image.open(uploaded_file)
                label, confidence = classify_image(image)

                if label != "Error":
                    primary_color = "#007BFF"
                    secondary_color = "#FF4136"
                    label_color = primary_color if label == "Matang" else secondary_color

                    st.sidebar.write(f"**Nama File:** {uploaded_file.name}")
                    st.sidebar.markdown(f"<h4 style='color: {label_color};'>Prediksi: {label}</h4>", unsafe_allow_html=True)

                    st.sidebar.write("**Confidence:**")
                    for i, class_name in enumerate(class_names):
                        st.sidebar.write(f"- {class_name}: {confidence[i] * 100:.2f}%")

                    custom_progress_bar(confidence, primary_color, secondary_color)

                    st.sidebar.write("---")
                else:
                    st.sidebar.error(f"Kesalahan saat memproses gambar {uploaded_file.name}: {confidence}")
            except Exception as e:
                st.sidebar.error(f"Kesalahan: {str(e)}")
    else:
        st.sidebar.error("Silakan unggah setidaknya satu gambar untuk diprediksi.")

if uploaded_files:
    st.write("### Preview Gambar")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)
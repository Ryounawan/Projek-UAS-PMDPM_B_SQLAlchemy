import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model(r'BestModel_MobileNet_SQLAlchemy.h5')
class_names = ['JamurEnoki', 'JamurKancing', 'JamurKuping']

def classify_image(image_path):
    try:
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])

        class_idx = np.argmax(result)
        confidence_scores = result.numpy()
        return class_names[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

def display_progress_bar(confidence, class_names):
    for i, class_name in enumerate(class_names):
        percentage = confidence[i] * 100
        st.sidebar.progress(int(percentage))
        st.sidebar.markdown(f'<p style="color: white;">{class_name}: {percentage:.2f}%</p>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://i.pinimg.com/736x/ef/c7/97/efc79797ab9b747eb7b25d3a2cddda03.jpg');
        background-size: cover;
        background-position: center center;
        color: white;
        font-family: 'Arial', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-image: url('https://i.pinimg.com/736x/ef/c7/97/efc79797ab9b747eb7b25d3a2cddda03.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        color: black;
    }

    h1 {
        color: #32CD32;
        text-align: center;
    }

    h2 {
        color: #FFD700;
        text-align: center;
    }

    footer {
        visibility: hidden;
    }

    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        color: #FFD700;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        Copyright © 2024 SQLAlchemy. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Prediksi Jenis Jamur: Enoki, Kancing, dan Kuping ")
st.write("Unggah Gambar (Beberapa diperbolehkan). terima kasih.")

uploaded_files = st.file_uploader("Unggah Gambar Jamur Anda (jpg, png, jpeg)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if st.sidebar.button("Prediksi"):
    if uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            label, confidence = classify_image(uploaded_file.name)
            
            if label != "Error":
                st.sidebar.markdown(f'<p style="color: black;"><b><i>Prediksi: {label}</i></b></p>', unsafe_allow_html=True)
                st.sidebar.write("Confidence:")
                display_progress_bar(confidence, class_names)
                st.sidebar.write("---")
            else:
                st.sidebar.error(f"Kesalahan saat memproses gambar {uploaded_file.name}: {confidence}")
    else:
        st.sidebar.error("Silahkan unggah setidaknya satu gambar untuk diprediksi")

if uploaded_files:
    st.write("### Preview Gambar")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name}", use_container_width =True)

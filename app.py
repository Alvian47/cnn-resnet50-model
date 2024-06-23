import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

label_models = ['Eczema', 'Melanocytic Nevi (NV)', 'Psoriasis Lichen Planus']
model = tf.keras.models.load_model("./m-cnn-2.h5")

output = None
uploaded_file = None
st.title("CNN Model RESNET50")

uploaded_file = st.file_uploader("Choose file")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    images = cv2.imdecode(file_bytes, -1)
    image_resized = cv2.resize(images, (256,256))
    images = np.expand_dims(image_resized, axis=0)
    pred = model.predict(images)
    output = label_models[np.argmax(pred)]

show = st.button("Show Image")
if show:
    if uploaded_file == None:
        st.write("Input picture first")
    else:
        st.image(uploaded_file, width=200)

if uploaded_file != None :
    st.markdown(f"<h1 style='color: white;'>{output}</h1>", unsafe_allow_html=True)

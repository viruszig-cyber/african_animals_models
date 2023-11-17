
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input


# loading the saved models

african_animals = pickle.load(open('african_animals_dataset.sav', 'rb'))
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

#heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))

#parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

map_dict = {0: 'dog',
            1: 'horse',
            2: 'elephant',
            3: 'butterfly',
            4: 'chicken',
            5: 'cat',
            6: 'cow',
            7: 'sheep',
            8: 'spider',
            9: 'squirrel'}



    # page title
st.title('Diabetes Prediction using ML') 
    
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = african_animals.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
















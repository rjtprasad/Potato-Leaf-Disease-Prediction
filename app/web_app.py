import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(
    page_title="Potato Leaf Disease Prediction"
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def load_model():
    model=tf.keras.models.load_model('potatoes.h5')
    return model

with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # Potato Leaf Disease Prediction
         """
         )
body = "Farmers who grow potatoes are facing lot of economic losses every year because of various " \
       "diseases that can happen to a potato plant. There are two common diseases known as early " \
       "blight and late blight early blight is caused by a fungus and " \
       "late blight is caused by a specific microorganism and if a farmer can detect these " \
       "diseases early and apply appropriate treatment then it can save lot of " \
       "waste and prevent the economic loss."
st.markdown(body, unsafe_allow_html=False)

file = st.file_uploader("", type=["jpg", "png"])

def import_and_predict(image_data, model):
        size = (256, 256)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis, ...]
        prediction = model.predict(img_reshape)
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['Early blight', 'Late blight', 'Healthy']
    string = "Prediction : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Healthy':
        st.success(string)
    else:
        st.warning(string)

html_link = """
    <a href="https://github.com/rjtprasad/" style="color:green;" target="_blank">Github</a><br>
    <a href="https://www.linkedin.com/in/prasadrajat/" style="color:green;" target="_blank">LinkedIn</a>
    """

st.markdown(html_link, unsafe_allow_html=True)
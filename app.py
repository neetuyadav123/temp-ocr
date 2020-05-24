import cv2
import numpy as np
from PIL import Image
from time import time
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


st.title("Devanagari Character Recognition")
st.sidebar.header("Machine Learning info")
st.sidebar.info('''For dependencies open project demonstration notebook.
	Further if you want to train for yourself then you need dataset also and may be 1 day of train time. I recommend google colab for train.''')

st.sidebar.subheader("Running options")
st.sidebar.info("upload a devanigiri hindi character")
st.sidebar.info("run video.py (webcam reqiured)")

st.sidebar.subheader("Model Creation")
st.sidebar.info('''4 days(7 hours /day at least)''')

st.sidebar.subheader("Project Training Process")
st.sidebar.markdown("""	
- Data Collection
- Data Manipulation
- Data Analysis
- Model Design
- Model Model Creation
- Model train/validation
- Model Analysis
- Model Delivery""")

st.sidebar.subheader("Recognition Model")
st.sidebar.markdown("""	
- Image Acquistion
- Image Preprocess
- Image Thresholdi
- Image Manipulati
- Text Detection
- Text Localizatio
- Text Recognition""")

@st.cache
def load_data():
    dataset = pd.read_csv("hello.csv")
    #X = dataset.iloc[:,:-1]
    Y_d = dataset.iloc[:,-1]
    return Y_d

def load_model(path ='model/model.h5'):
    return tf.keras.models.load_model(path)

num_pixels = 1024
num_classes = 46
img_width = 32
img_height = 32
img_depth = 1

with st.spinner("Please wait loading data"):
    Y_d = load_data()

#X_images = X.values.reshape(X.shape[0], img_width, img_height)
binencoder = LabelBinarizer()
Y = binencoder.fit_transform(Y_d)


uploaded_file = st.file_uploader("Select an image with Hindi Text",type=['png','jpg'])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.header("Uploaded Image")
    st.image(image.resize((256,256),Image.BILINEAR), caption='Uploaded Image.')
    image = image.resize((img_width,img_height),Image.LIBIMAGEQUANT)
    image = np.array(image) / 255.0
    st.header("the image as numpy array")
    st.write(image)

    
    with st.spinner("Please wait loading Model"):
        model = load_model()
        img = image.reshape(num_pixels).reshape(1,-1)
        st.write(image.reshape(num_pixels).reshape(-1,1).shape)
        y_pred = model.predict(img)
        st.header("Predicted numpy data")
        st.write(y_pred.shape)
        result = binencoder.inverse_transform(y_pred)
        st.write(result)


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from PIL import Image,ImageFilter

num_pixels = 1024
num_classes = 46
img_width = 32
img_height = 32
img_depth = 1

def prediction(img):
    model = load_model()
    y_pred = model.prediction([img])
    return y_pred

def bw_image(img_path, thresh = 200):
    img = Image.open(img_path)
    fn = lambda x : 255 if x > thresh else 0
    r = img.convert('L').point(fn, mode='1')
    r.save('uploads/img.jpg')
    return r

def load_model(model_path = "model/model.h5"):
    return tf.keras.models.load_model(model_path)

if __name__ == "__main__":
    img = bw_image('tes1.png')
    img.show()
from PIL import Image
import io
from flask import render_template, Flask, redirect, url_for, request
import random
import os
import numpy as np
from keras.applications.mobilenet import MobileNet 
from keras.preprocessing import image
import tensorflow as tf
import streamlit as st

from tensorflow.keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.models import model_from_json
from keras import backend as K
import sys
from io import BytesIO 
# from StringIO import StringIO

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
app = Flask(__name__)

SKIN_CLASSES = {
0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
1: 'Basal Cell Carcinoma',
2: 'Benign Keratosis',
3: 'Dermatofibroma',
4: 'Melanoma',
5: 'Melanocytic Nevi',
6: 'Vascular skin lesion'

}

@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/uploaded', methods = ['GET', 'POST'])
def upload_file(): 
    if request.method == 'POST':
        f = request.files['file']
        path='static/data/'+f.filename
        f.save(path)
        with open('modelnew.json', 'r', encoding = 'utf-8') as j_file:
            loaded_json_model = j_file.read()
        with open(path, 'rb') as img_file:
            img_content = img_file.read()
        model = model_from_json(loaded_json_model)
        model.load_weights('modelnew.h5')
        img1 = Image.open(BytesIO(img_content))
        img1 = img1.resize((224, 224))
        img1 = np.array(img1)
        img1 = img1 / 255.0
        prediction = model.predict(np.expand_dims(img1, axis=0))
        pred = np.argmax(prediction)
        disease = SKIN_CLASSES[pred]
        accuracy = prediction[0][pred]
        K.clear_session()
    return render_template('uploaded.html', title='Success', predictions=disease, acc=accuracy*100, img_file=f.filename)

if __name__ == "__main__":
    app.run(debug=True)
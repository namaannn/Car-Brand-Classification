from __future__ import division,print_function
from flask import Flask,send_from_directory,redirect,url_for,request,render_template
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
import sys
import glob
import re

app=Flask(__name__)
model=load_model('resnet.h5')
def model_predict(file_path,model):
    img=image.load_img(file_path,target_size=(254,254))
    x=image.img_to_array(img)
    x=x/255.
    x=np.expand_dims(x,axis=0)
    pred=model.predict(x)
    preds=np.argmax(pred,axis=1)
    if preds==0:
        preds="This car is Audi"
    elif preds==1:
        preds="This car is Lamborghini"
    else:
        preds="This car is Mercedes Benz"
    
    return preds
@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        preds=model_predict(file_path,model)
        result=preds
        return result
    return None
if __name__=='__main__':
    app.run(debug=True)
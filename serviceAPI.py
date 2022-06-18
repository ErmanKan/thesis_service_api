# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:30:25 2022

@author: erman
"""
import cv2
import numpy as np
import pickle
import os
import base64
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

BLUE_UPPER = np.array([110,255,255])
BLUE_LOWER = np.array([83,0,0])
train_labels = ["1 ml(0.5ml kan)", "2 ml(1ml kan)", "3 ml(1.5ml kan)", "4 ml(2ml kan)", "5 ml(2.5ml kan)", "6 ml(3ml kan)" , "7 ml(3.5ml kan)", "8 ml(4ml kan)" ,"9 ml(4.5ml kan)", "10 ml(5ml kan)", "11 ml(5.5ml kan)", "12 ml(6ml kan)"]

def getImage(image_path):
    image = cv2.imread(image_path,1)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    return image

def getMask(mask_lower, mask_upper,image):
    mask = cv2.inRange(image, mask_lower, mask_upper)
    return mask

def applyMask(image,mask):
    masked_image = cv2.bitwise_and(image, image, mask)
    return masked_image

def hu_moments(image):
    hu_moments = cv2.HuMoments(cv2.moments(image)).flatten()
    return hu_moments

def generate_histogram(image):
    hist = cv2.calcHist([image],[0],None, [32],[0,256])
    cv2.normalize(hist,hist)
    return hist.flatten()

def processImage(file):
        img = getImage(file)
        mask = getMask(BLUE_LOWER, BLUE_UPPER, img)
        masked = applyMask(img,mask)
        blur = cv2.GaussianBlur(masked,(9,9),0)
        gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        
        feature = hu_moments(gray)
        hist = generate_histogram(gray)
        
        plt.plot(hist)
        plt.title('Histogram Of The Image (Grayscale)')
        plt.xlabel('Bins')
        plt.ylabel('Counts')

        plt.savefig(file.replace(".","a") + 'plot.jpg')
        plt.clf()

        global_feature = np.hstack([hist,feature])
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaled_feature = scaler.fit_transform(global_feature.reshape(-1,1))
        
        prediction = loaded_model.predict(rescaled_feature.reshape(1,-1))[0]
        print("Prediction is : ", prediction)
        
        return prediction
    

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/getClass', methods=['POST'])
def classify():
    print("Received a post request!")
    if(request.method == 'POST'):
        image_file = request.files['image']
        filename = secure_filename(image_file.filename)
        joined = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace("\\","/")
       
        image_file.save(joined)
        print("Saved the image file!")
        print("Prediction ")
        prediction = processImage(joined)

        hist_path = joined.replace(".","a") + 'plot.jpg'
        encoded = base64.b64encode(open(hist_path, "rb").read()).decode('ascii')

        return jsonify(message = 'Success',prediction = prediction, hist = str(encoded))
        
        
    
@app.route('/healthcheck',methods=['GET'])
def healthcheck():
    return jsonify(message = "Healthy")

if __name__=="__main__":
    loaded_model = pickle.load(open("rfc_model.sav", 'rb'))
    app.run(port = 8080,debug = True)
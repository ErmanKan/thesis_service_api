# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:30:25 2022

@author: erman
"""
import cv2
import numpy as np
import pickle
import os
import mahotas
import base64
import pandas as pd
from flask import Flask, request, jsonify
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

BLUE_UPPER = np.array([110,255,255])
BLUE_LOWER = np.array([83,0,0])
train_labels = ["1 ml(0.5ml kan)", "2 ml(1ml kan)", "3 ml(1.5ml kan)", "4 ml(2ml kan)", "5 ml(2.5ml kan)", "6 ml(3ml kan)" , "7 ml(3.5ml kan)", "8 ml(4ml kan)" ,"9 ml(4.5ml kan)", "10 ml(5ml kan)", "11 ml(5.5ml kan)", "12 ml(6ml kan)"]

def getImage(image_path):
    image = cv2.imread(image_path,1)
    return image

def getMask(mask_lower, mask_upper,image):
    mask = cv2.inRange(image, mask_lower, mask_upper)
    mask = cv2.bitwise_not(mask)
    return mask

def applyMask(image,mask):
    masked_image = np.zeros_like(image)
    masked_image = cv2.bitwise_and(image, image, mask = mask)
    return masked_image

def hu_moments(image):
    hu_moments = cv2.HuMoments(cv2.moments(image)).flatten()
    return hu_moments

def generate_histogram(image):
    hist = cv2.calcHist([image],[0],None, [32],[0,256])
    cv2.normalize(hist,hist)
    return hist.flatten()

def getLaplacianHist(laplacian):
    hist, edges = np.histogram(laplacian,bins = 12,range = (-50,50))
    return hist

def getBloodHistogram(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    RED_MIN = np.array([0, 50, 50])
    RED_MAX = np.array([10, 255, 255])
    mask0 =  cv2.inRange(img, RED_MIN, RED_MAX)
 
    
    RED_HIGH_MIN = np.array([170,50,50])
    RED_HIGH_MAX = np.array([180,255,255])
    mask1 =  cv2.inRange(img, RED_HIGH_MIN, RED_HIGH_MAX)
    
    threshed = mask0 + mask1
    masked_img = applyMask(img, threshed)
    img = cv2.cvtColor(masked_img,cv2.COLOR_HSV2BGR)
    return generate_histogram(img)
    
def getHaralickFeatures(img):
    haralick = mahotas.features.haralick(img).mean(axis=0)
    return haralick

def processImage(file):

        img = getImage(file)
        imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
        mask = getMask(BLUE_LOWER, BLUE_UPPER, imgHSV)
        maskedBGR = applyMask(img,mask)
        maskedHSV = cv2.cvtColor(maskedBGR, cv2.COLOR_BGR2HSV)
        
        blur = cv2.GaussianBlur(maskedBGR,(9,9),0)
        blur = cv2.cvtColor(maskedHSV, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        
        laplacian = cv2.Laplacian(gray,cv2.CV_64F, ksize = 1)
        
        bloodHist = getBloodHistogram(maskedBGR)
        laplacianHist = getLaplacianHist(laplacian)
        haralickFeatures = getHaralickFeatures(gray)
        huMomentsFeature = hu_moments(gray)
        hist = generate_histogram(gray)
        
        global_feature = np.concatenate((hist,huMomentsFeature,haralickFeatures,laplacianHist,bloodHist), axis = 0)

        prediction = loaded_model.predict(global_feature.reshape(1,-1))[0]
        
        plt.plot(hist)
        plt.title('Histogram Of The Image (Grayscale)')
        plt.xlabel('Bins')
        plt.ylabel('Counts')

        plt.savefig(file.replace(".","a") + 'plot.jpg')
        plt.clf()
        return train_labels[prediction]


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/getClass', methods=['POST'])
def classify():
    print("*******************************")
    print("Received a post request!")
    if(request.method == 'POST'):
        image_file = request.files['image']
        filename = secure_filename(image_file.filename)
        joined = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace("\\","/")
        print("Filename: ", joined)
        image_file.save(joined)
        print("Saved the image file!")
        
        prediction = processImage(joined)
        print("Prediction is: ", prediction)
        hist_path = joined.replace(".","a") + 'plot.jpg'
        print("Histogram path: ", hist_path)
        encoded = base64.b64encode(open(hist_path, "rb").read()).decode('ascii')

        return jsonify(message = 'Success',prediction = prediction, hist = str(encoded))
        
        
    
@app.route('/healthcheck',methods=['GET'])
def healthcheck():
    return jsonify(message = "Healthy")

if __name__=="__main__":
    loaded_model = pickle.load(open("rfc_model.sav", 'rb'))
    app.run(port = 8080,debug = True)
    
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle
import mahotas
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score
from sklearn.pipeline import Pipeline


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
    



###############################################################
"""
BLUE_UPPER = np.array([110,255,255])
BLUE_LOWER = np.array([83,0,0])
label_count = 12
images_per_class = 15
label = 0
train_path = "Images\\train"
test_path = "Images\\test"
train_labels = ["1 ml(0.5ml kan)", "2 ml(1ml kan)", "3 ml(1.5ml kan)", "4 ml(2ml kan)", "5 ml(2.5ml kan)", "6 ml(3ml kan)" , "7 ml(3.5ml kan)", "8 ml(4ml kan)" ,"9 ml(4.5ml kan)", "10 ml(5ml kan)", "11 ml(5.5ml kan)", "12 ml(6ml kan)"]

global_features = []
labels = []

for label in train_labels:
    dir = os.path.join(train_path, label)
    print(dir)
    current_label = label
    
    for count in range(1, images_per_class+1):
        folder = dir + "\\" + "image_"
        file = folder  + "("+ str(count) + ")" + ".jpg"
        print(file)
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
        
        global_feature = np.hstack([hist,huMomentsFeature,haralickFeatures,laplacianHist,bloodHist])
        labels.append(current_label)
        global_features.append(global_feature)
        
    print("processed folder: {}".format(current_label))
    
targetNames = np.unique(labels)
encoder = LabelEncoder()
target = encoder.fit_transform(labels)

print("target labels: {}".format(target))
print("target labels shape: {}".format(target.shape))

df_data = pd.DataFrame(global_features)
df_labels = pd.DataFrame(target)

df_data.to_csv("data.csv",index = False)
df_labels.to_csv("labels.csv", index = False)
"""
################################################################

seed = 0
results = []
names   = []
models = []

models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed)))

data = pd.read_csv("data.csv")
labels = pd.read_csv("labels.csv")
(trainData, testData, trainLabels, testLabels) = train_test_split(data.values, labels.values.ravel(), test_size=0.33, random_state=seed)

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, trainData, trainLabels, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print("MODEL: ",name)
    print("CV_RESULTS:")
    print(msg)
    model.fit(trainData,trainLabels)
    y_prediction = model.predict(testData)
    print("\nClassification report of model:\n",classification_report(y_prediction,testLabels))
    print("Accuracy score:",100*accuracy_score(y_prediction,testLabels))
    print()

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

pipeline = Pipeline([('scaler', StandardScaler()),('rf', RandomForestClassifier(n_estimators=100, random_state=seed))])

pipeline.fit(trainData, trainLabels)

filename = 'rfc_model.sav'
pickle.dump(pipeline, open(filename, 'wb'))
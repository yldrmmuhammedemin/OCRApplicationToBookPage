import cv2 as cv
import numpy as np
from skimage.feature import hog
import pickle
import random
from sklearn.svm import SVC

#For noise of words
def controlword(cc):
    x, y, w, h = cv.boundingRect(cc)
    area = cv.contourArea(cc)
    if w <10  or h <12  or area <90:
        return False
    else:
        return True
#For noise of rows
def controlrow(cc):
    x, y, w, h = cv.boundingRect(cc)
    area = cv.contourArea(cc)
    if w < 10 or h<10 or area<120 :
        return False
    else:
        return True
#For noise of cahars
def controlchar(contour):
    x, y, w, h = cv.boundingRect(contour)
    area = cv.contourArea(contour)
    if  area <10:
        return False
    else:
        return True
##Preprocessing step##
img=cv.imread('Page.jpeg')
img=cv.resize(img,(750,1000))
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(imgray,80,255,cv.THRESH_BINARY_INV)[1]
file=open("OCR.txt","w")
#rows
kernelrow=cv.getStructuringElement(cv.MORPH_RECT,(30,1))
kernelrow2=cv.getStructuringElement(cv.MORPH_RECT,(30,9))
kernelrow3=cv.getStructuringElement(cv.MORPH_RECT,(40,4))
threshrow=cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernelrow)
threshrow=cv.morphologyEx(threshrow,cv.MORPH_OPEN,kernelrow2)
threshrow=cv.morphologyEx(threshrow,cv.MORPH_CLOSE,kernelrow)
threshrow=cv.morphologyEx(threshrow,cv.MORPH_DILATE,kernelrow3)
#words
kernelwrd = cv.getStructuringElement(cv.MORPH_RECT, (6,2))
threshwrd= cv.dilate(thresh, kernelwrd)
#chars
threshchar=thresh.copy()
##
##Reading Dataset
with open('BookDataset','rb') as cdata:
    chardata=pickle.load(cdata)
cdata.close()

random.shuffle(chardata)
features=[]
labels=[]

for feature,label in chardata:
    features.append(feature)
    labels.append(label)
##
#SVM Model
svm_model = SVC(kernel='poly', gamma='auto', C=10)
svm_model.fit(features, labels)
#
#Row sorting
rowcontours,hier= cv.findContours(threshrow, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
rowcontours = [contour for i, contour in enumerate(rowcontours) if controlrow(contour)]
rowboxes = []
for i,rowcntr in enumerate(rowcontours):
    xr,yr,wr,hr = cv.boundingRect(rowcntr)
    rowboxes.append((xr, yr, wr, hr))
def takeSecond(elem):
    return elem[1]
rowboxes.sort(key=takeSecond)
for rowbox in rowboxes:
    xr = rowbox[0]
    yr = rowbox[1]
    wr = rowbox[2]
    hr = rowbox[3]
    row = threshwrd[yr:yr+hr, xr:xr+wr]
    bboxes = []
    ctrs, hier = cv.findContours(row, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    ctrs = [contour for i, contour in enumerate(ctrs) if controlword(contour)]
    for i, cntr in enumerate(ctrs):
        x,y,w,h = cv.boundingRect(cntr)
        bboxes.append(((x+xr),(y+yr),w,h))
    def takeFirst(elem):
        return elem[0]
    bboxes.sort(key=takeFirst)
    file.write("\n")
#Words sorting
    for box in bboxes:
        xb = box[0]
        yb = box[1]
        wb = box[2]
        hb = box[3]
        roiw = threshchar[yb:yb + hb, xb:xb + wb]
        charbox=[]
        ctrs,hier= cv.findContours(roiw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        ctrs = [contour for i, contour in enumerate(ctrs) if controlchar(contour)]
        for i, cntr in enumerate(ctrs):
            x, y, w, h = cv.boundingRect(cntr)
            charbox.append(((x+xb), (y+yb) , w, h))
        def takeFirst(elem):
            return elem[0]
        charbox.sort(key=takeFirst)
        file.write(" ")
#Char sorting
        for cbox in charbox:
            xc = cbox[0]
            yc = cbox[1]
            wc = cbox[2]
            hc = cbox[3]
            #Test data prepration with HOG feature
            roic = threshchar[yc:yc + hc, xc:xc + wc]
            roic=255-roic
            roic=cv.resize(roic,(60,60))
            hogfv, hogimage = hog(roic, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True)
            hogimage = np.array(hogimage).flatten()
            hogimage=hogimage.reshape((1,3600))
            ##
            #Prediction the test data
            prediction = svm_model.predict(hogimage)
            categories = ['a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e','E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n','N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't', 'T', 'u', 'U', 'v', 'V','w','W', 'x', 'X', 'y', 'Y', 'z', 'Z']
            pred=prediction[0]
            file.write(categories[pred])##writing the prediction step
file.close()
file=open("OCR.txt","r+")
print(file.read())
cv.waitKey(0)

import cv2
import numpy as np
from keras.models import load_model
import tensorflow
from PIL.Image import *
model=load_model('gender_final_a.h5')
model_age=load_model('age_kaggle_b.h5')
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
source=cv2.VideoCapture(0)
##source=cv2.imread('queen.jpeg')
labels_dict={0:'MEN',1:'WOMEN'}
labels_dict_age={0:'Child-0-14',1:'Youth-15:25',2:'Adult:26-40',3:'Old:40-60',4:'Seniors:60+'}
color_dict={0:(0,255,0),1:(0,0,255)}
color_dict_age={0:(0,255,0),1:(0,0,255),2:(255,0,0),3:(255,255,0),4:(255,255,255)}
while(True):
    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ##gray=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(224,224))
        resized_age=cv2.resize(face_img,(32,32))
        resized= cv2.cvtColor(resized,cv2.COLOR_GRAY2RGB)
        resized_age= cv2.cvtColor(resized_age,cv2.COLOR_GRAY2RGB)
        ##normalized=resized/255.0
        ##normalized_age=resized_age/255.0
        reshape=np.expand_dims(resized,axis=0)
        normalized=reshape/255.0
        reshape_age=np.expand_dims(resized_age,axis=0)
        normalized_age=reshape_age/255.0
        ##reshaped=np.reshape(normalized,(1,100,100,3))
        result=model.predict(normalized)
        result_a=model_age.predict_classes(normalized_age)
        ##print(result_a)

        result=np.argmax(result)
        ##result_a=np.argmax(result_a)
    
        print(result_a)

        if(result==0):
            label=0
        elif(result==1):
            label=1
        
        if(result_a==0):
            label_a=0
        elif(result_a==1):
            label_a=1
        elif(result_a==2):
            label_a=2
        elif(result_a==3):
            label_a=3
        elif(result_a==4):
            label_a=4


        ##label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict_age[label_a],-1)
        cv2.putText(img, labels_dict[label], (x, y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(img, labels_dict_age[label_a], (x, y+50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),3)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindow
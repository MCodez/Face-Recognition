# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:22:05 2017

@author: LALIT ARORA
"""

import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("trainer\\trainingData.yml")
id=0
name="NONE"
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        print(id)
        if id==1:
            name="LALIT"
        elif id==2:
            name="Modiji"
        elif id==3:
            name="Obama"
        else:
            name="NOONE"
        cv2.putText(img,name,(x,y+h),font,6,(0,0,255),4)
    cv2.imshow("Face",img)
    if(cv2.waitKey(1)==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
    
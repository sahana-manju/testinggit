# -*- coding: utf-8 -*-
"""
Created on Sat May 15 20:26:37 2021

@author: Admin
"""
import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image as im

img=cv2.imread('E:/landmark/1.jpeg')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


faces = hog_face_detector(gray,1)
for face in faces:

    face_landmarks = dlib_facelandmark(gray, face)
    coordinates=[]
    for n in range(0, 68):
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y
        coordinates.append(x)
        coordinates.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), 1)

    
    
    cv2.rectangle(img, pt1=(150,249),pt2=(180,259),color=(0,255,0), thickness=1)
    
    cv2.imshow("Face Landmarks", img)

    key = cv2.waitKey(100000)
    if key == 27:
        break

plt.imshow(img)

cv2.destroyAllWindows()
data = im.fromarray(rect)
data.save('new.jpg')
test_image=tf.keras.preprocessing.image.load_img('E:/landmark/new.jpg',target_size=(60,60))
test_image=tf.keras.preprocessing.image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
model=tf.keras.models.load_model('wrinkle_detection.h5')
result=model.predict(test_image)
if result[0][0]>result[0][1]:
    print('no wrinkles found')
else:
    print('wrinkles percentage',result[0][1]*100,'%')

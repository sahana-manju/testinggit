# -*- coding: utf-8 -*-
"""
Created on Sun May 16 12:42:44 2021

@author: Admin
"""
import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image as im
import matplotlib.pyplot as plt


img = cv2.imread("E:/landmark/1.jpeg")
mask = np.zeros(img.shape[0:2], dtype=np.uint8)
img[mask == 255] = (255, 255, 255)
points = np.array([[[140, 240],
                          [164, 222],
                          [192,274],
                          [206,267],
                          [225,265],
                          [241,260],  
                          [240,280],
                          [226,280],
                          [212,280],
                          [194,280],
                          [184,280],
                         ]], np.int32)
#     #method 1 smooth region
cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
#     #method 2 not so smooth region
cv2.fillPoly(mask, points, (0))
res = cv2.bitwise_and(img,img,mask = mask)
rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
cv2.imshow("Cropped", cropped )
cv2.waitKey(0)
cv2.destroyAllWindows()
data = im.fromarray(cropped)
data.save('1.jpg')
test_image=tf.keras.preprocessing.image.load_img('E:/shapes/nowrinkles/nw223.jpg',target_size=(60,60))
test_image=tf.keras.preprocessing.image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
model=tf.keras.models.load_model('wrinkle_detection.h5')
result=model.predict(test_image)
if result[0][0]>result[0][1]:
    print('no wrinkles found')
else:
    print('wrinkles percentage',result[0][1]*100,'%')


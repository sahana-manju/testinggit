# -*- coding: utf-8 -*-
"""
Created on Sun May 23 13:44:39 2021

@author: Admin
"""

import sys
import os
import dlib
import glob
#from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt


img=cv2.imread('E:/landmark/eyew.jpeg')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hog_face_detector = dlib.get_frontal_face_detector()

predictor_path = 'shape_predictor_81_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


# frame = cv2.flip(frame, 1)
dets = detector(img, 0)

leftCheekPoints = [1, 2, 3, 4, 48, 31, 30, 29, 28, 1]
rightCheekPoints = [15, 14, 13, 12, 11, 54, 35, 30, 29, 28, 15]
foreHeadPoints = [75, 21, 27, 22, 74, 79, 73, 72, 80, 71, 70, 69, 75]
chinPoints = [6, 7, 8, 9, 10, 55, 56, 57, 58, 59, 6]


for k, d in enumerate(dets):
    shape = predictor(img, d)
    landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
    
    mask = np.zeros(img.shape[:2], np.uint8)
   
    leftCheek = np.array([ [shape.parts()[num].x, shape.parts()[num].y] for num in leftCheekPoints ])
    rightCheek = np.array([ [shape.parts()[num].x, shape.parts()[num].y] for num in rightCheekPoints ])
    foreHead = np.array([ [shape.parts()[num].x, shape.parts()[num].y] for num in foreHeadPoints ])
    chin = np.array([ [shape.parts()[num].x, shape.parts()[num].y] for num in chinPoints ])
    
    
#     cv2.polylines(mask, [np.array(leftCheek)], True, (255,255,255), cv2.LINE_AA)
#     cv2.polylines(mask, [np.array(rightCheek)], True, (255,255,255), cv2.LINE_AA)
#     cv2.polylines(mask, [np.array(foreHead)], True, (255,255,255), cv2.LINE_AA)
#     cv2.polylines(mask, [np.array(chin)], True, (255,255,255), cv2.LINE_AA)
    
    cv2.drawContours(mask, [np.array(leftCheek)], -1, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.drawContours(mask, [np.array(rightCheek)], -1, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.drawContours(mask, [np.array(foreHead)], -1, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.drawContours(mask, [np.array(chin)], -1, (255, 255, 255), -1, cv2.LINE_AA)
    
    dst = cv2.bitwise_and(img, img, mask=mask)

    


plt.figure(figsize=(10, 10))
plt.imshow(dst)
plt.show()
print(dst)
cv2.imwrite("Slice.jpg", dst) # To save Image
edges = cv2.Canny(dst,100,130)
l1=[]
for i in range(667):
    for j in range(1000):
        if edges[i][j]>0:
            if [i,j] not in leftCheek:
                l1.append([i,j])
                cv2.circle(img, (j,i), radius=0, color=(0, 0, 255), thickness=-1)
print(l1)
product=gray*edges

number_of_edges = np.count_nonzero(edges)
print(number_of_edges)
if number_of_edges>3000:
    print('wrinkles found')
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
else:
    print('no wrinkle found')
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

plt.figure(figsize=(10, 10))


plt.show()
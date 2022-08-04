# -*- coding: utf-8 -*-
"""
Created on Sun May 16 13:49:17 2021

@author: Admin
"""


import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt

import os
from PIL import Image as im

img=cv2.imread('E:/landmark/old.png')
edges = cv2.Canny(img,100,130)
l1=[]
for i in range(600):
    for j in range(419):
        if edges[i][j]>0:
            l1.append([i,j])
            cv2.circle(img, (j,i), radius=0, color=(0, 0, 255), thickness=-1)
plt.imshow(img)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:48:59 2019

@author: arun
"""

import os
from PIL import Image

import numpy as np

import cv2

videodims = (854,480)
fourcc = cv2.VideoWriter_fourcc(*'MPEG')    
video = cv2.VideoWriter("testgirl.mp4",fourcc, 24,videodims)



mypath = "/home/arun/RANet/datasets/DAVIS/JPEGImages/480p/kite-surf"
maskpath = "/home/arun/RANet/predictions/RANet_Video_17val/kite-surf/"

jpg_onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
mask_onlyfiles = [os.path.join(maskpath, f) for f in os.listdir(maskpath) if os.path.isfile(os.path.join(maskpath, f))]

for k , image_file1 in enumerate(mask_onlyfiles):
    
    image_file = format(k, '05d') + ".jpg"
    print(image_file)
    image = Image.open(mypath + "/" + image_file)
    
    mask = Image.open(maskpath + "/" + image_file.split('.')[0] + ".png")
    maskL = mask.convert('L')
    mask = mask.convert('RGB')
    

    blend_image = Image.blend(image, mask, 0.5)
    blend_image = Image.composite(image, blend_image,  maskL)
    
    save_path = maskpath + "/" + image_file.split('.')[0] + ".jpg"
    video.write(cv2.cvtColor(np.array(blend_image), cv2.COLOR_RGB2BGR))
    blend_image.save(save_path)
    
    
    if k == 128:
        break
    
video.release()
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:20:27 2025

@author: Yash

Detects the edges of the coins using Canny-Edge Detector.
"""

import cv2

# Defining the path to read the image from
path = "images/image1.jpg"

# Reading the image
img = cv2.imread(path)

# Converting the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Defining the name of the image
img_name = "Coins"


# Using Canny-Edge Detector to detect the edges in the image
# Setting parameter values 
t_lower = 100  # Lower Threshold 
t_upper = 300  # Upper threshold 
  
# Applying the Canny Edge filter 
edge = cv2.Canny(gray_img, t_lower, t_upper)

# Display the image
cv2.imshow(img_name, edge)
cv2.waitKey(0)
cv2.destroyAllWindows()


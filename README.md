# VR_Assignment1_-Yash-Sengupta-_-IMT2022532-

## 1.py
* The code is used to detect the edges of coins using Canny-Edge Detector.
* The threshold values are set as `t_lower=100` and `t_upper=300` which successfully detects all the edges.
* ![Screenshot 2025-02-16 125920](https://github.com/user-attachments/assets/7c663222-7aca-4079-a477-88374ef3c290)

## 2.py
* This code is used to perform regeon based segmentation of coins. Then it uses contour detection to detect the boundaries of the different objects and then we use the number of boundaries detected to count the number of coins.
* It uses Watershed Algorithm.
* ![Screenshot 2025-02-16 130155](https://github.com/user-attachments/assets/fba76f4c-acc3-4033-84cd-dde73cd0a813)
* The algorithm basically provides regeon based masks for every coin and uses it to identify contours. We can access these masks, to view the mask of every detected image.
* ![Screenshot 2025-02-16 130350](https://github.com/user-attachments/assets/57c74b75-eaf9-43f2-ae24-7a6c118cebdc)

## 3.py
* This code is used to stitch a sequence of images to form a panaroma. Standard methods and algorithms from OpenCV library are used to form the panaroma.
* Output: ![Screenshot 2025-02-16 130531](https://github.com/user-attachments/assets/fb984264-529f-4d5c-b51d-119b75631f6b)
* The problem is while stiching these images, we can see that for some images, the scale is not matched properly.


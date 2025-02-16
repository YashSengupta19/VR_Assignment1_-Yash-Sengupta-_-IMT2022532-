# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:17:37 2025

@author: Yash

Image Stitching to create a panaroma.
"""

import cv2
import numpy as np

# Load all images
image_paths = [
    "images/sticth1.jpg",
    "images/sticth2.jpg",
    "images/sticth3.jpg",
    "images/sticth4.jpg"
]

images = [cv2.imread(path) for path in image_paths]

# Check if all images are loaded correctly
for i, img in enumerate(images):
    if img is None:
        print(f"Error: Could not load image {image_paths[i]}")
        exit()

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors for all images
keypoints_descriptors = [sift.detectAndCompute(img, None) for img in images]
keypoints, descriptors = zip(*keypoints_descriptors)

# Use FLANN-based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Function to stitch two images using homography
def stitch_images(img1, img2, kp1, des1, kp2, des2):
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    MIN_MATCH_COUNT = 10
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp img1 to align with img2
        stitched_img = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))

        # Overlay img2 onto the stitched result
        stitched_img[0:img2.shape[0], 0:img2.shape[1]] = img2

        return stitched_img
    else:
        print(f"Not enough matches found ({len(good_matches)}/{MIN_MATCH_COUNT}).")
        return None

# Stitch images sequentially (left to right)
stitched = images[0]
for i in range(1, len(images)):
    stitched = stitch_images(stitched, images[i], keypoints[i-1], descriptors[i-1], keypoints[i], descriptors[i])
    if stitched is None:
        print(f"Error stitching image {i+1}")
        exit()

# Resize the final result to fit the screen
screen_width, screen_height = 2000, 3000  # Adjust as needed
h, w = stitched.shape[:2]
scale_factor = min(screen_width / w, screen_height / h)
stitched_resized = cv2.resize(stitched, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

# Display the final stitched image
cv2.imshow("Stitched Panorama", stitched_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the final stitched output
cv2.imwrite("stitched_output.jpg", stitched)
print("Final stitched image saved as stitched_output.jpg")



import cv2
import numpy as np

# Load the segmented SAR image (binary mask)
segmented_image = cv2.imread('C:/Users/akira/codewithsenpai/oil/oil/predicted_img_1071_jpg.rf.93efb42e8ca2978e9d6d99d49c737669.jpg', cv2.IMREAD_GRAYSCALE)

# Define the threshold (assume oil-affected areas are white: 255)
oil_pixels = np.sum(segmented_image == 255)  # Count pixels with value 255

print(f"Number of oil-affected pixels: {oil_pixels}")
pixel_size = 1.5 * 1.5  # Area of one pixel in m²
oil_area = oil_pixels * pixel_size

print(f"Oil-affected area: {oil_area} square meters")


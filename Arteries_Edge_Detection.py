import os
import numpy as np
from skimage import io, color, filters, morphology
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Load the image
curr_dir = os.getcwd()

# Change file name to see other pages.
whole_img = os.path.join(curr_dir, "final_data\\train\\images\\H653A_0001_patch_12_4.png")

# Load the original image
original_image = Image.open(whole_img)

# Convert the image to a NumPy array
original_image = np.array(original_image)

# Apply Gaussian blur to the original image
blurred_image = filters.gaussian(original_image, sigma=1)

# Convert the blurred image to grayscale
blurred_gray = color.rgb2gray(blurred_image)

# Apply a lower threshold to segment the objects of interest more aggressively
threshold = 0.58  # Adjust the threshold as needed
mask = blurred_gray > threshold

# Remove small connected components (objects) from the mask
min_object_size = 10 # Adjust the size threshold as needed
cleaned_mask = morphology.remove_small_objects(mask, min_size=min_object_size, connectivity=2)

# Create a copy of the original image
image_with_contours = original_image.copy()

# Find contours of the objects in the mask using OpenCV
contours, _ = cv2.findContours(np.uint8(cleaned_mask) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the copy of the original image with a thinner line
cv2.drawContours(image_with_contours, contours, -1, (255, 0, 0), 1)  # Red color, 1-pixel thickness

# Display or save the final image with contours
plt.imshow(image_with_contours)
plt.axis('off')
plt.show()

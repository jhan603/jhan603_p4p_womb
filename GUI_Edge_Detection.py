import os
import numpy as np
from skimage import io, color, filters, morphology, measure
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Scale, Button, Label, Entry
import cv2

# Function to process the image and update the displayed image
def process_image():
    # Get the threshold and min_object_size values from user input
    threshold_value = float(threshold_entry.get())
    min_object_size_value = int(min_object_size_entry.get())

    # Apply Gaussian blur to the original image
    blurred_image = filters.gaussian(original_image, sigma=1)

    # Convert the blurred image to grayscale
    blurred_gray = color.rgb2gray(blurred_image)

    # Apply a lower threshold to segment the objects of interest
    mask = blurred_gray > threshold_value

    # Remove small connected components (objects) from the mask
    cleaned_mask = morphology.remove_small_objects(mask, min_size=min_object_size_value, connectivity=2)

    # Create a copy of the original image
    image_with_contours = original_image.copy()

    # Find contours of the objects in the mask using OpenCV
    contours, _ = cv2.findContours(np.uint8(cleaned_mask) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the copy of the original image with a thinner line
    cv2.drawContours(image_with_contours, contours, -1, (255, 0, 0), 1)  # Red color, 1-pixel thickness

    # Resize the processed image to make it larger
    large_image = cv2.resize(image_with_contours, (800, 600))

    # Convert the processed image to a format compatible with Tkinter
    image_with_contours_pil = Image.fromarray(large_image)
    image_with_contours_tk = ImageTk.PhotoImage(image=image_with_contours_pil)

    # Update the image widget in the GUI
    image_label.config(image=image_with_contours_tk)
    image_label.image = image_with_contours_tk

# Function to select an image from the file dialog
def select_image():
    global original_image
    image_path = filedialog.askopenfilename()
    if not image_path:
        return  # User canceled file selection
    original_image = Image.open(image_path)
    original_image = np.array(original_image)

    # Display the selected file name in the GUI
    selected_file_label.config(text=f"Selected File: {os.path.basename(image_path)}")

# Create the main application window
app = tk.Tk()
app.title("Object Detection and Contour Drawing")
app.geometry("800x600")  # Set the window size to 800x600

# Create a button to open the file dialog
open_button = Button(app, text="Select Image", command=select_image)
open_button.pack()

# Create a label and entry for the threshold
threshold_label = Label(app, text="Threshold Value: (Recommended between 0.5-0.6)")
threshold_label.pack()
threshold_entry = Entry(app)
threshold_entry.pack()

# Create a label and entry for the minimum object size
min_object_size_label = Label(app, text="Min Object Size: (Recommended between 5-20")
min_object_size_label.pack()
min_object_size_entry = Entry(app)
min_object_size_entry.pack()

# Create a button to process the image
process_button = Button(app, text="Process Image", command=process_image)
process_button.pack()

# Create a label to display the selected file name
selected_file_label = Label(app, text="Selected File:")
selected_file_label.pack()

# Create an image label to display the processed image
image_label = Label(app)
image_label.pack()

app.mainloop()

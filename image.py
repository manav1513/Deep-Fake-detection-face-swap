import os
import cv2
import numpy as np

# Path to the single image files (replace with actual file paths)
image_file_A = 'WhatsApp Image 2024-10-09 at 3.17.15 PM.jpeg'  # Example file path
image_file_B = 'WhatsApp Image 2024-10-09 at 3.17.15 PM.jpeg'  # Replace with actual second image file path

# Check if the files exist
if not os.path.isfile(image_file_A):
    raise FileNotFoundError(f"The file '{image_file_A}' does not exist. Please provide a valid file path.")
if not os.path.isfile(image_file_B):
    raise FileNotFoundError(f"The file '{image_file_B}' does not exist. Please provide a valid file path.")

# Load the images
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")
    return cv2.resize(img, (128, 128))  # Resize to 128x128

# Load the images for Face A and Face B
image_A = load_image(image_file_A)
image_B = load_image(image_file_B)

# Convert to float32 for model input
image_A = np.array(image_A, dtype=np.float32) / 255.0
image_B = np.array(image_B, dtype=np.float32) / 255.0

# Ensure both images have the shape (128, 128, 3)
print("Image A shape:", image_A.shape)
print("Image B shape:", image_B.shape)

# Now you can proceed with your autoencoder model or further processing

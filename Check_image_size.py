import tensorflow as tf
from PIL import Image

# Load an image using TensorFlow
image = tf.keras.preprocessing.image.load_img("CT_KIDNEY_DATASET_Normal_Cyst_Tumor_Stone/CT_KIDNEY_DATASET_Normal_Cyst_Tumor_Stone/Cyst/Cyst- (1).jpg")

# Convert image to numpy array
image_array = tf.keras.preprocessing.image.img_to_array(image)

# Get the shape of the image
height, width, channels = image_array.shape
print("Image shape:", height, "x", width, "x", channels)
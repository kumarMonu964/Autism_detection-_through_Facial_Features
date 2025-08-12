import numpy as np
from PIL import Image

def preprocess_image(image_file, target_size=(224, 224)):
    image = Image.open(image_file).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # shape: (1, 224, 224, 3)
    return image_array

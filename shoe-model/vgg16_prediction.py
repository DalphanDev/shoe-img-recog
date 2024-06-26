import os
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import custom_object_scope

# Load the label encoder from the file
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
    
# Custom layer to replicate grayscale to 3 channels
class GrayToRGB(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.repeat(inputs, 3, axis=-1)

# Load the saved model
# model = load_model("shoeSilhouetteModel_VGG16.h5")
with custom_object_scope({'GrayToRGB': GrayToRGB}):
    model = load_model("shoeSilhouetteModel_VGG16.h5")

# Define the image size
image_size = (224, 224)

# Load a webp image
image_path = os.path.join(os.getcwd(), "shoe-model/prediction/img6.webp")
img = Image.open(image_path).convert('L')  # Convert to grayscale
img = img.resize(image_size)  # Resize the image
img_array = np.array(img)  # Convert the image to a numpy array

# Preprocess the image
img_array = img_array.astype('float32')
img_array /= 255.0
img_array = np.expand_dims(img_array, axis=0)
img_array = np.expand_dims(img_array, axis=-1)

# Make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Invert the label encoding to obtain the original label
predicted_label = label_encoder.inverse_transform(predicted_class)[0]

print("Predicted label:", predicted_label)

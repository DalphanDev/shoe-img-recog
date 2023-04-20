import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import custom_object_scope
from PIL import ImageOps


def get_class_labels(train_dir):
    class_labels = sorted(os.listdir(train_dir))
    label_map = {index: label for index, label in enumerate(class_labels)}
    return label_map


def preprocess_image(image_path, target_size):
    img = image.load_img(
        image_path, target_size=target_size, color_mode='grayscale')
    img = ImageOps.autocontrast(img)
    img = img_to_array(img)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


class GrayToRGB(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.repeat(inputs, 3, axis=-1)


# Load the saved model
with custom_object_scope({'GrayToRGB': GrayToRGB}):
    model = load_model(os.path.join(
        os.getcwd(), "silhouette-model/silhouetteModel_VGG16.h5"))

# Define the image size
image_size = (299, 299)

# Load an image
image_path = os.path.join(
    os.getcwd(), "silhouette-model/prediction/nb993.jpg")
preprocessed_img = preprocess_image(image_path, (299, 299))

# Create the label map
train_dir = os.path.join(os.getcwd(), "silhouette-model/train")
label_map = get_class_labels(train_dir)

# Make a prediction
predictions = model.predict(preprocessed_img)

# Find the predicted class
predicted_class = np.argmax(predictions, axis=-1)

# Find the confidence level (probability) of the predicted class
confidence_level = np.max(predictions) * 100

# Get the label corresponding to the predicted class
predicted_label = label_map[predicted_class[0]]
print("Predicted label:", predicted_label)
print("Confidence level: {:.2f}%".format(confidence_level))

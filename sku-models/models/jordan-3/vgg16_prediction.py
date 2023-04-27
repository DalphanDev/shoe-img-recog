import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def squarify_image(img: Image.Image) -> Image.Image:
    target_size = max(img.width, img.height)
    # Create a new image with the same size and white background
    square_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    square_image.paste(
        img, ((target_size - img.width) // 2, (target_size - img.height) // 2)
    )
    return square_image


def get_class_labels(train_dir):
    class_labels = sorted(os.listdir(train_dir))
    label_map = {index: label for index, label in enumerate(class_labels)}
    return label_map


def preprocess_image(image_path, target_size, data_generator):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = data_generator.standardize(img_array)
    return preprocessed_img


# Load the saved model
model = load_model(
    os.path.join(os.getcwd(), "sku-models/models/jordan-3/jordan3_VGG16.h5")
)

# Define the image size
image_size = (224, 224)

# Define the ImageDataGenerator for prediction
prediction_data_gen = ImageDataGenerator(rescale=1.0 / 255)

# Load an image
image_path = os.path.join(os.getcwd(), "sku-models/models/jordan-3/prediction/img.jpg")

# Preprocess the image using the prediction ImageDataGenerator
preprocessed_img = preprocess_image(image_path, image_size, prediction_data_gen)

# Create the label map
train_dir = os.path.join(os.getcwd(), "sku-models/models/jordan-3/train")
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

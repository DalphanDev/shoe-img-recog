from flask import Flask, request, jsonify
import os
import io
import requests
import numpy as np
import tensorflow as tf
import validators
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import custom_object_scope

app = Flask(__name__)
keylist = ["4dd306d5-eba6-4e39-93e9-64fd12f6eec4"]
# Flask app is started


# First setup our functions for our models.
class GrayToRGB(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.repeat(inputs, 3, axis=-1)


def squarify_image_RGB(img: Image.Image) -> Image.Image:
    target_size = max(img.width, img.height)
    # Create a new image with the same size and white background
    square_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    square_image.paste(
        img, ((target_size - img.width) // 2, (target_size - img.height) // 2)
    )
    return square_image


def squarify_image_grayscale(img: Image.Image) -> Image.Image:
    target_size = max(img.width, img.height)
    # Create a new image with the same size and white background
    square_image = Image.new("L", (target_size, target_size), 255)
    square_image.paste(
        img, ((target_size - img.width) // 2, (target_size - img.height) // 2)
    )
    return square_image


def get_class_labels(train_dir):
    class_labels = sorted(os.listdir(train_dir))
    label_map = {index: label for index, label in enumerate(class_labels)}
    return label_map


def preprocess_image_grayscale(image_path, target_size, data_generator):
    img = image.load_img(image_path, target_size=target_size, color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = data_generator.standardize(img_array)
    return preprocessed_img
    return img


def preprocess_image_rgb(image_path, target_size, data_generator):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = data_generator.standardize(img_array)
    return preprocessed_img


def is_valid_url(url):
    if validators.url(url):
        return True
    else:
        return False


# Define the image size for the model
image_size = (224, 224)

# Define the ImageDataGenerator for prediction
prediction_data_gen = ImageDataGenerator(rescale=1.0 / 255)

# Define our model paths
model_paths = []

# Append each model path to the list
model_paths.append(
    os.path.join(os.getcwd(), "silhouette-model/silhouetteModel_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/jordan-1-high/jordan1High_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/jordan-1-mid/jordan1Mid_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/jordan-1-low/jordan1Low_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/jordan-2/jordan2_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/jordan-3/jordan3_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/jordan-4/jordan4_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/jordan-5/jordan5_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/jordan-5-low/jordan5Low_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/jordan-6/jordan6_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/jordan-6-low/jordan6Low_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/jordan-11/jordan11_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/jordan-11-low/jordan11Low_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/nb-990v3/nb990v3_VGG16.h5")
)
model_paths.append(os.path.join(os.getcwd(), "sku-models/models/nb-993/nb993_VGG16.h5"))
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/nike-af1-low/nikeAF1Low_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/nike-am1/nikeAM1_VGG16.h5")
)
model_paths.append(
    os.path.join(os.getcwd(), "sku-models/models/nike-dunk-low/nikeDunkLow_VGG16.h5")
)

print(f"Loading {len(model_paths)} models...")

# Load all of our models
models = {}
label_maps = {}

for model_path in model_paths:
    # Replace backslashes with forward slashes
    model_path = model_path.replace("\\", "/")

    # Extract the model name from the path
    rawModelName = os.path.splitext(os.path.basename(model_path))[0]
    cleanModelName = rawModelName.split("_")[0]
    directory = os.path.dirname(model_path)
    parent_directory_name = os.path.basename(directory)
    if "silhouette" in model_path:
        # Load the saved model
        with custom_object_scope({"GrayToRGB": GrayToRGB}):
            silhouetteModel = load_model(
                os.path.join(os.getcwd(), "silhouette-model/silhouetteModel_VGG16.h5")
            )
    else:
        # Create a variable with the model name and assign it the path value
        models[parent_directory_name] = load_model(model_path)

    print(f"Loaded {parent_directory_name}...")

    # Create the label maps for each model
    train_dir = os.path.dirname(model_path) + "/train"
    label_map = get_class_labels(train_dir)
    label_maps[parent_directory_name] = label_map
    print(f"Created {parent_directory_name} label map...")

print(models)


@app.route("/api", methods=["POST"])
def predict():
    # Error handling
    if not request.is_json:
        return jsonify({"error": "Not using application/json content-type header"}), 400
    if not request.json:
        return jsonify({"error": "Invalid request body"}), 400
    if "key" not in request.json:
        return jsonify({"error": "No key property in JSON body"}), 400
    if request.json["key"] not in keylist:
        return jsonify({"error": "Invalid key"}), 400
    if "img" not in request.json:
        return jsonify({"error": "No img property in JSON body"}), 400
    if is_valid_url(request.json["img"]) == False:
        return jsonify({"error": "Invalid img url"}), 400

    data = request.get_json()
    print(data)
    # Download the image from the URL
    response = requests.get(data["img"])
    img = Image.open(io.BytesIO(response.content)).convert("RGBA")

    # # Save the image to the server
    # img.save("/tmp/temp_image.jpg")  # Change this to use a uuid instead of temp_image

    # Preprocess the image for the silhouette model
    # Create a new image with the same size and white background
    white_background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    # Paste the webp image using its alpha channel as a mask
    white_background.paste(img, mask=img.split()[3])
    rgb_image = white_background.convert("RGB")

    # Convert the image to grayscale
    grayscale_image = rgb_image.convert("L")

    silhouetteImg = squarify_image_grayscale(grayscale_image)

    # Define the directory
    temp_dir = os.path.join(os.getcwd(), "tmp")

    # Create the directory if it does not exist
    os.makedirs(temp_dir, exist_ok=True)

    # Temporarily save silhouette image

    silhouetteImgPath = os.path.join(os.getcwd(), "tmp\\temp_sil_image.jpg")

    silhouetteImg.save(silhouetteImgPath)

    processedSilhouetteImg = preprocess_image_grayscale(
        silhouetteImgPath, image_size, prediction_data_gen
    )

    # Preprocess the image for the sku model
    # Create a new image with the same size and white background
    white_background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    # Paste the webp image using its alpha channel as a mask
    white_background.paste(img, mask=img.split()[3])
    rgb_image = white_background.convert("RGB")

    skuImg = squarify_image_RGB(rgb_image)

    # Temporarily save sku image

    skuImgPath = os.path.join(os.getcwd(), "tmp\\temp_sku_image.jpg")

    skuImg.save(skuImgPath)

    processedSkuImg = preprocess_image_rgb(skuImgPath, image_size, prediction_data_gen)

    # Make a silhouette prediction
    silhouette_predictions = silhouetteModel.predict(processedSilhouetteImg)

    # Find the predicted class
    silhouette_predicted_class = np.argmax(silhouette_predictions, axis=-1)

    # Find the confidence level (probability) of the predicted class
    silhouette_confidence_level = np.max(silhouette_predictions) * 100

    # Get the label corresponding to the predicted class
    silhouette_predicted_label = label_maps["silhouette-model"][
        silhouette_predicted_class[0]
    ]
    print("Predicted label:", silhouette_predicted_label)
    print("Confidence level: {:.2f}%".format(silhouette_confidence_level))

    # Select the model based on the silhouette prediction

    if silhouette_predicted_label not in models:
        return (
            jsonify(
                {
                    "silhouette": {
                        "prediction": silhouette_predicted_label,
                        "confidence": round(silhouette_confidence_level, 2),
                    },
                    "sku": {
                        "prediction": "No SKU prediction available",
                        "confidence": "No SKU prediction available",
                    },
                }
            ),
            200,
        )

    sku_model = models[silhouette_predicted_label]

    # Make a sku prediction
    sku_predictions = sku_model.predict(processedSkuImg)

    # Find the predicted class
    sku_predicted_class = np.argmax(sku_predictions, axis=-1)

    # Find the confidence level (probability) of the predicted class
    sku_confidence_level = np.max(sku_predictions) * 100

    # Get the label corresponding to the predicted class
    sku_predicted_label = label_maps[silhouette_predicted_label][sku_predicted_class[0]]

    return (
        jsonify(
            {
                "silhouette": {
                    "prediction": silhouette_predicted_label,
                    "confidence": round(silhouette_confidence_level, 2),
                },
                "sku": {
                    "prediction": sku_predicted_label,
                    "confidence": round(sku_confidence_level, 2),
                },
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run()

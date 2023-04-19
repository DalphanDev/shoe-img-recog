import os
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

# Define the image size
image_size = (224, 224)

# Define the directories
train_dir = os.path.join(os.getcwd(), "shoe-model/train")
test_dir = os.path.join(os.getcwd(), "shoe-model/test")
val_dir = os.path.join(os.getcwd(), "shoe-model/val")

# Create a label encoder object
label_encoder = LabelEncoder()

# Load and preprocess the training images
train_images = []
train_labels = []

print("Importing training image dataset...")

for label in os.listdir(train_dir):
    label_dir = os.path.join(train_dir, label)
    for filename in os.listdir(label_dir):
        img = Image.open(os.path.join(label_dir, filename))
        img = img.resize(image_size)
        img = img.convert('L')  # convert to grayscale
        img_array = np.array(img)
        train_images.append(img_array)
        train_labels.append(label)

# Convert the training images and labels to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Load and preprocess the testing images
test_images = []
test_labels = []

print("Importing testing image dataset...")

for label in os.listdir(test_dir):
    label_dir = os.path.join(test_dir, label)
    for filename in os.listdir(label_dir):
        img = Image.open(os.path.join(label_dir, filename))
        img = img.resize(image_size)
        img = img.convert('L')  # convert to grayscale
        img_array = np.array(img)
        test_images.append(img_array)
        test_labels.append(label)

# Convert the testing images and labels to numpy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Load and preprocess the validation images
val_images = []
val_labels = []

print("Importing validation image dataset...")

for label in os.listdir(val_dir):
    label_dir = os.path.join(val_dir, label)
    for filename in os.listdir(label_dir):
        img = Image.open(os.path.join(label_dir, filename))
        img = img.resize(image_size)
        img = img.convert('L')  # convert to grayscale
        img_array = np.array(img)
        val_images.append(img_array)
        val_labels.append(label)

# Convert the validation images and labels to numpy arrays
val_images = np.array(val_images)
val_labels = np.array(val_labels)

# Define the data augmentation
data_generator = ImageDataGenerator(horizontal_flip=True)

# Apply the data augmentation to the training images
train_images_augmented = data_generator.flow(train_images.reshape(-1, 224, 224, 1), batch_size=len(train_images), shuffle=False).next()

# Display the first n images
n = 5
for i in range(n):
    plt.imshow(train_images_augmented[i], cmap='gray')
    plt.title(f"Label: {train_labels[i]}")
    plt.show()

# Convert the augmented training images to tensors
train_images_augmented = tf.convert_to_tensor(train_images_augmented, dtype=tf.float32)

# Load the VGG-16 base model
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Define the custom head for the model
head_model = base_model.output
head_model = layers.GlobalAveragePooling2D()(head_model)
head_model = layers.Dense(512, activation='relu')(head_model)
head_model = layers.Dropout(0.5)(head_model)
head_model = layers.Dense(29, activation='softmax')(head_model)

# Custom layer to replicate grayscale to 3 channels
class GrayToRGB(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.repeat(inputs, 3, axis=-1)

# Combine the base model and the custom head
model = models.Model(inputs=base_model.input, outputs=head_model)

# Add GrayToRGB layer to the beginning of the model
model = models.Sequential([
    GrayToRGB(input_shape=(image_size[0], image_size[1], 1)),
    model
])

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()

# Fit the label encoder to the training labels
label_encoder.fit(train_labels)

# Save the label encoder to a file
with open("label_encoder_2.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Encode the training labels
train_labels = label_encoder.transform(train_labels)

# Encode the validation labels
val_labels = label_encoder.transform(val_labels)

# Encode the test labels
test_labels = label_encoder.transform(test_labels)

# Format the labels for the model
train_labels = to_categorical(train_labels, num_classes=29)
val_labels = to_categorical(val_labels, num_classes=29)
test_labels = to_categorical(test_labels, num_classes=29)

# Train the model with the augmented training images
history = model.fit(train_images_augmented, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
model.save("shoeSilhouetteModel_VGG16_2.h5")

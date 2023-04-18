import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Define the image size
image_size = (299, 299)

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

# Convert the training images to tensors
train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)

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

# Convert the testing images to tensors
test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)

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
        

# Convert the training images and labels to numpy arrays

val_images = np.array(val_images)
val_labels = np.array(val_labels)

# Convert the validation images to tensors
val_images = tf.convert_to_tensor(val_images, dtype=tf.float32)

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(29, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()

# Fit the label encoder to the training labels
label_encoder.fit(train_labels)

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

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
model.save("shoeSilhouetteModel.h5")

# if (test_acc > 0.95):
#     model.save("shoeSilhouetteModel.h5")
# else:
#   print('Model accuracy is too low to save. Try adding more images to the training set.')
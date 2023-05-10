import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Define the image size
image_size = (224, 224)

# Define the shoe model to train for
shoeModel = "jordan-5"

# Define the directories
train_dir = os.path.join(os.getcwd(), "sku-models\\models\\" + shoeModel + "\\train")
test_dir = os.path.join(os.getcwd(), "sku-models\\models\\" + shoeModel + "\\test")
val_dir = os.path.join(os.getcwd(), "sku-models\\models\\" + shoeModel + "\\val")

# Define the number of classes
num_classes = 10

# Define the number of epochs
num_epochs = 100

# Load the VGG-16 base model
base_model = tf.keras.applications.VGG16(
    weights="imagenet", include_top=False, input_shape=(image_size[0], image_size[1], 3)
)

# Define the custom head for the model
head_model = base_model.output
head_model = layers.GlobalAveragePooling2D()(head_model)
head_model = layers.Dense(512, activation="relu")(head_model)
head_model = layers.Dropout(0.5)(head_model)
head_model = layers.Dense(num_classes, activation="softmax")(head_model)


# Combine the base model and the custom head
model = models.Model(inputs=base_model.input, outputs=head_model)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Print the model summary
model.summary()

# Define the data generators
train_data_gen = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=(0.8, 1.2),
    height_shift_range=0.2,
    rescale=1.0 / 255,
)
train_generator = train_data_gen.flow_from_directory(
    train_dir, target_size=image_size, batch_size=32, class_mode="categorical"
)

val_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
val_generator = val_data_gen.flow_from_directory(
    val_dir, target_size=image_size, batch_size=32, class_mode="categorical"
)

test_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_data_gen.flow_from_directory(
    test_dir, target_size=image_size, batch_size=32, class_mode="categorical"
)

# Add Early Stopping callback
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Train the model with the augmented training images
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[early_stopping],
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print("Test accuracy:", test_acc)
model.save(
    os.path.join(os.getcwd(), "sku-models\\models\\" + shoeModel + "\\jordan5_VGG16.h5")
)

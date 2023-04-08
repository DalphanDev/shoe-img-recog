import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras import layers

# Define the input shape
input_shape = (128, 128, 3)

# Define the number of classes
num_classes = 3

# Define the directories
train_dir = os.path.join(os.getcwd(), "cow-natural/Train")
test_dir = os.path.join(os.getcwd(), "cow-natural/Test")
val_dir = os.path.join(os.getcwd(), "cow-natural/Val")

# Define the model architecture
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()

# Define the data generators
train_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_data_gen.flow_from_directory(train_dir, target_size=input_shape[:2], batch_size=32, class_mode='categorical')

val_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_generator = val_data_gen.flow_from_directory(val_dir, target_size=input_shape[:2], batch_size=32, class_mode='categorical')

test_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_data_gen.flow_from_directory(test_dir, target_size=input_shape[:2], batch_size=32, class_mode='categorical')

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

if (test_acc > 0.95):
    model.save("cowNaturalModel.h5")
else:
  print('Model accuracy is too low to save. Try adding more images to the training set.')
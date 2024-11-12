import os
import numpy as np
import keras
from keras.preprocessing import image
from keras.utils import to_categorical
from keras import models, layers
from sklearn.model_selection import train_test_split
import cv2  # For image loading and resizing

# Location of images
path = 'training_images/'
image_files = os.listdir(path)

print('Training image folder successfully found.')

# Load and process images manually
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            img = cv2.resize(img, (150, 150))  # Resize to 150x150
            img = img / 255.0  # Normalize pixel values to [0, 1]
            images.append(img)
            # Assuming that the folder name indicates the label (e.g., "pest" or "non_pest")
            label = 1 if "pest" in filename else 0
            labels.append(label)
    return np.array(images), np.array(labels)

# Load training images
train_images, train_labels = load_images_from_folder("training_images/")
validation_images, validation_labels = load_images_from_folder("validation_images/")

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Define your model
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # For binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=10)

# Save the trained model
model.save('/home/robocode/anaconda3/training_code/radar_pest_model.h5')

# Evaluate the model on test images
test_images = os.listdir("data/head_ct_slices/test")

for file_name in test_images:
    path = "data/head_ct_slices/test/" + file_name
    test_img = image.load_img(path, target_size=(150, 150))
    img = image.img_to_array(test_img)
    img /= 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    print(f"Prediction: {prediction[0]}")

    if prediction[0] < 0.5:
        print(file_name + " image is pest")
    else:
        print(file_name + " image not pest")
    print('\n')


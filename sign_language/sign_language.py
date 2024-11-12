import os
import numpy as np
import keras
from keras.preprocessing import image
from keras.utils import to_categorical
from keras import models, layers
from sklearn.model_selection import train_test_split
import cv2  # For image loading and resizing

# Root directory path
root_path = '/home/robocode/anaconda3/training_code/sign_language'

print('Training image folder successfully found.')

# Map each letter to a unique integer label (e.g., A -> 0, B -> 1, ..., Z -> 25)
letter_to_label = {chr(i): i - 65 for i in range(65, 91)}  # Mapping for letters A-Z

# Load and process images manually
def load_images_from_folder(folder):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        label = letter_to_label.get(subfolder.upper(), None)  # Get label for letter
        if label is None:
            continue
        for filename in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                img = cv2.resize(img, (150, 150))  # Resize to 150x150
                img = img / 255.0  # Normalize pixel values to [0, 1]
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load training and validation images
train_images, train_labels = load_images_from_folder(os.path.join(root_path, "train"))
validation_images, validation_labels = load_images_from_folder(os.path.join(root_path, "validation"))

# One-hot encode labels
train_labels = to_categorical(train_labels, num_classes=26)  # 26 classes for A-Z
validation_labels = to_categorical(validation_labels, num_classes=26)

# Split the dataset if additional validation data is not available
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
    layers.Dense(26, activation='softmax')  # For multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=10)

# Save the trained model
model_path = os.path.join(root_path, 'asl_model.h5')
model.save(model_path)

# Evaluate the model on test images
test_images_path = os.path.join(root_path, "test")
test_images = os.listdir(test_images_path)
for file_name in test_images:
    img_path = os.path.join(test_images_path, file_name)
    test_img = image.load_img(img_path, target_size=(150, 150))
    img = image.img_to_array(test_img)
    img /= 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_label = np.argmax(prediction[0])  # Get the index with the highest probability
    predicted_letter = chr(predicted_label + 65)  # Convert to corresponding letter

    print(f"File: {file_name}, Predicted letter: {predicted_letter}\n")


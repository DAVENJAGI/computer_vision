import os
import random
# import matplotlib.pyplot as plt

# Location of images
path = 'training_images/'
image_files = os.listdir(path)

print('Training image folder successfully found.')

'''
This is where we import tensorflow and train with our data
'''
import keras
from keras.preprocessing.image import ImageDataGenerator

# Normalize images: normalization typically entails scaling values to between 0 and 1.
                  # Images are made of pixels with varying intensities from 0 to 255.
                  # So normalizing the images (dividing each pixel by 255) scales the pixel values to between 0 and 1.
                  # This range of values is more effective for a neural network to learn from.
train_data_generator = ImageDataGenerator(rescale=1/255)
validate_data_generator = ImageDataGenerator(rescale=1./255)

# Indicate the location of our train and validation set of images.
train_folder = "training_images/"
validation_folder = "validation_images/"

print("Both training and validation folders found")

''' Provide images to the model in batches using the train_data_generator,
the images are resized to 150 by 150 for their length and width, and fed to in batches if 10 images.
'''
train_generator = train_data_generator.flow_from_directory(
        train_folder,
        target_size=(150, 150),
        batch_size=10,
        class_mode="binary")

''' Provides the model with a validation set of images using the validate_data_generator
the image resizing for the validation generator is the same as for the image training algorithm
'''
validate_generator = validate_data_generator.flow_from_directory(
        validation_folder,
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary')

'''Here we employ convolutional neural networks to learn from the data,
We have layers, input output and hidden layer.
'''

import keras
from keras import models, layers

model = models.Sequential([

    layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(),

    # Layer 2
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    # Layer 3
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    # Layer 4
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),


    # Fully-Connected Layer
    # Flatten the learned features into a 1-dimensional array of values.
    # Pass the learned features to the fully-connected layer (Dense layer in TensorFlow) for prediction.
    layers.Flatten(),
    layers.Dense(256, activation='relu'), # 'relu' helps the model focus on what's most important and speeds up training
    layers.Dense(1, activation='sigmoid') # 'sigmoid' ensures that the prediction will be between 0 and 1,
                                          # interpreted as the probability of being in the positive class ("normal").
])


'''
Compile the Built Model
'binary_crossentropy' (this is a binary task) tells the model how to measure its loss (error)
'adam' indicates how the model will adjust its weights, while learning, to decrease its loss (error)
'''
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics = ['accuracy'])

'''Here we begin to train the model with the data collected'''
history = model.fit(
      train_generator,  # Provides the training images to the model.
      validation_data = validate_generator, # Provides the validation set of images to the model to assess training.
      epochs=10)   

model.save('/home/robocode/anaconda3/training_code/radar_pest_model.h5')


"""using test data to figure out if its accurate or not,
here is where we figure out if it's a pest or not.
"""
from keras.preprocessing import image
import numpy as np
import os

#  The location of our test (unseen) images, used to evaluate the model's predictions
test_images = os.listdir("data/head_ct_slices/test")

# Make predictions on the test images
for file_name in test_images:
    path = "data/head_ct_slices/test/" + file_name
    test_img = image.load_img(path, target_size = (150,150))
    img = image.img_to_array(test_img)
    img /= 255.0    # normalization to between 0 and 1).
    img = np.expand_dims(img, axis = 0)

    prediction = model.predict(img)

    print(f"Prediction: {prediction[0]}")

  # Output the file name based on probability of the model either being greater than or less tha 0.5
    if prediction[0] < 0.5:
        print(file_name + " image is pest")

    else:
        print(file_name + " image not pest")
    print('\n')


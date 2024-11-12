import cv2
import numpy as np
import keras
from keras.models import load_model
import serial
import time
import os

model = load_model('asl_model.h5')

arduino = serial.Serial('/dev/ttyUSB0', 9600)
time.sleep(2)

cap = cv2.VideoCapture(0)

def preprocess_frame(frame):
    frame = cv2.resize(frame, (150, 150))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        input_frame = preprocess_frame(frame)
        prediction = model.predict(input_frame)
        
        predicted_label = np.argmax(prediction[0])
        predicted_letter = chr(predicted_label + 65)
        
        # Check if the predicted letter is A
        if predicted_letter == 'A':
            print("Letter 'A' detected! Activating buzzer.")
            arduino.write(b'1')
        else:
            arduino.write(b'0')
            
        cv2.putText(frame, f"Predicted Letter: {predicted_letter}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("The code is running successfully")

finally:
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()


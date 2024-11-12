import cv2
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
import serial
import time


model = load_model('/radar_pest_model.h5')

arduino = serial.Serial('/dev/ttyUSB0', 9600)
time.sleep(2)

cap = cv2.VideoCapture(0)

def preprocess_frame(frame):
    
    frame = cv2.resize(frame, (150, 150))
    frame = frame / 255.0 #normalizing frame
    frame = np.expand_dims(frame, axis=0)
    return frame

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        #pre processing the frame and predicting
        input_frame = preprocess_frame(frame)
        prediction = model.predict(input_frame)

        if prediction[0] < 0.5:
            print("Pest detected!")
            arduino.write(b'1') 
        else:
            arduino.write(b'0')  

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("the code is running successfully")

finally:
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()


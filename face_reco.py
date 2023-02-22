import cv2
import os
import numpy as np

# Get face with webcam
cam = cv2.VideoCapture(0)
cam.set(3, 1080) 
cam.set(4, 1080)

# Load face detector
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create face data directory and face train output directory
faceDir = 'faces_data'
faceTrainOutputDir = 'face_train_output'
if not os.path.exists(faceDir):
    os.makedirs(faceDir)
if not os.path.exists(faceTrainOutputDir):
    os.makedirs(faceTrainOutputDir)

# Load trained face recognizer
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    faceRecognizer.read(os.path.join(faceTrainOutputDir, 'train.xml'))
except cv2.error:
    print('Could not read the trained face recognizer model.')

# Set font for displaying text
font = cv2.FONT_HERSHEY_SIMPLEX

# Set parameters for face detection
minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

# Load names from a text file
names_file = open('names.txt', 'r')
names = names_file.readlines()
names_file.close()

# Strip newlines from names
names = [name.strip() for name in names]
names[0] = 'Unknow'

while True:
    # Capture frame from webcam
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceDetector.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=5, 
        minSize=(round(minWidth), round(minHeight))
    )

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Recognize the face
        id, confidence = faceRecognizer.predict(gray[y:y+h, x:x+w])
        if confidence <= 50:
            nameId = names[id]
            confidenceText = "{0}%".format(round(100-confidence))
        else:
            nameId = "Unknown"
            confidenceText = "{0}%".format(round(100-confidence))
        
        # Display the name and confidence level on the frame
        cv2.putText(frame, nameId, (x+5, y-5), font, 1, (255, 255, 255), 2)
            
    # Display the frame
    cv2.imshow('frame', frame)
    keyClose = cv2.waitKey(1) & 0xFF
    if keyClose == ord('q') & keyClose == 27:
        break
    
cam.release()
cam.destroyAllWindows()

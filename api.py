import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from flask import Response
from flask_restful import Resource, Api
import sys
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
port = 5100
CORS(app)

# get face with webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480)

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetector  = cv2.CascadeClassifier('haarcascade_eye.xml')
faceDataIndex = 1
faceDir = 'faces_data'
faceTrainOutputDir = 'face_train_output'
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()


@app.route('/api/record', methods=['POST'])
def record_face():
    global faceDataIndex
    #faceId = request.form['id']
    lastId = 0  # id terakhir yang digunakan
    for subdir, dirs, files in os.walk(faceDir):
        for file in files:
            path = os.path.join(subdir, file)
            fileName, fileExt = os.path.splitext(file)
            faceId = int(fileName.split('.')[1])
            if faceId > lastId:
                lastId = faceId

    nextId = lastId + 1
    faceId = nextId
    nama = request.form['nama']

    # Open names.txt in append mode
    with open('names.txt', 'a') as file:
        # Write the name to the file
        file.write(nama + '\n')

    while True:
        retV, frame = cam.read()
        frame = cv2.flip(frame, 1)
        abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5)

        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            filename = 'face.'+str(faceId)+'.'+str(faceDataIndex)+'.jpg'
            cv2.imwrite(os.path.join(faceDir+'/'+filename), frame)
            faceDataIndex += 1

            rolAbuAbu = abuAbu[y:y+h, x:x+w]
            rolWarna = frame[y:y+h, x:x+w]
            eyes = eyeDetector.detectMultiScale(rolAbuAbu)
            for (xe, ye, we, he) in eyes:
                cv2.rectangle(rolWarna, (xe, ye), (xe+we, ye+he), (0, 0, 255), 1)

        cv2.imshow('frame', frame)
        # cv2.imshow('frame2', abuAbu)
        keyClose = cv2.waitKey(1) & 0xFF
        if keyClose == ord('q') & keyClose == 27:
            break
        elif faceDataIndex > 30:
            break

    return jsonify({'message': 'Data wajah berhasil direkam.'})


@app.route('/api/train', methods=['POST'])
def train_face():
    faces, IDs = getImageLabel(faceDir)

    try:
        faceRecognizer.train(faces, np.array(IDs))
        faceRecognizer.write(faceTrainOutputDir+'/train.xml')
        return jsonify({'message': 'Pelatihan wajah berhasil dilakukan.'})
    except Exception as e:
        return jsonify({'message': 'Error: ' + str(e)})


def getImageLabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    faceIds = []

    for imgPath in imagePaths:
        PIlimg = Image.open(imgPath).convert('L') #convert ke greyscale
        imgNum = np.array(PIlimg, 'uint8')
        faceId = int(os.path.split(imgPath)[-1].split('.')[1])
        faces = faceDetector.detectMultiScale(imgNum)

        for (x, y, w, h) in faces:
            faceSamples.append(imgNum[y:y+h, x:x+w])
            faceIds.append(faceId)
    return faceSamples, faceIds

@app.route('/api/predict', methods=['POST'])
def predict_face():
    try:
        imgFile = request.files['file']
        img = Image.open(imgFile).convert('L') #convert ke greyscale
        imgNum = np.array(img, 'uint8')
        
        train_face()

        faces = faceDetector.detectMultiScale(imgNum)
        for (x, y, w, h) in faces:
            face, conf = faceRecognizer.predict(imgNum[y:y+h, x:x+w])

            with open('names.txt') as file:
                names = file.readlines()
                name = names[face-1].strip()

            imgNum = cv2.rectangle(imgNum, (x, y), (x+w, y+h), (0, 255, 255), 2)
            imgNum = cv2.putText(imgNum, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        _, imgEncoded = cv2.imencode('.jpg', imgNum)
        imgBytes = imgEncoded.tobytes()
        
        return jsonify({
            'message': "Wajah ditemukan",
            'data': name,
        })

        #return Response(response=imgBytes, status=200, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'message': 'Error: ' + str(e)})

    
if sys.argv.__len__() > 1:
    port = sys.argv[1]
print("Api running on port : {} ".format(port))

class topic_tags(Resource):
    def get(self):
        return {'hello': 'world world'}

api.add_resource(topic_tags, '/')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port)
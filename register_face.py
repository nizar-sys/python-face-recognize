import cv2, os, numpy as np
from PIL import Image

# get face with webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480)

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetector  = cv2.CascadeClassifier('haarcascade_eye.xml')
facesDataIndex = 1
faceDir = 'faces_data'
faceTrainOutputDir = 'face_train_output'
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

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
nama = input('Masukkan nama Anda: ')


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
        filename = 'face.'+str(faceId)+'.'+str(facesDataIndex)+'.jpg'
        
        cv2.imwrite(os.path.join(faceDir+'/'+filename), frame)
        facesDataIndex += 1
        
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
    elif facesDataIndex >30:
        break
    
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

print('Menyimpan data')
faces, IDs = getImageLabel(faceDir)

try:
    faceRecognizer.train(faces, np.array(IDs))
    faceRecognizer.write(faceTrainOutputDir+'/train.xml')
    print('success')
except Exception as e:
    print('error:', e)
    
cam.release()
cv2.destroyAllWindows()
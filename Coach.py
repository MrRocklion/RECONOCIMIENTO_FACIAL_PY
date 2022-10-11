import cv2 as cv
import os
import imutils as imu
import numpy as np

datapath = "C:/Users/David/Documents/UNIVERSIDAD/PYTHON/PROYECTO_RECONOCIMIENTO_FACIAL/Data"

Lista = os.listdir(datapath)
print('Lista de personas', Lista)

labels = []
faceData = []
label = 0

for nameDir in Lista:
    PersonaPath = datapath + '/' + nameDir

    for Filename in os.listdir(PersonaPath):
        print('Rostros', nameDir + '/'+Filename)
        labels.append(label)
        faceData.append(cv.imread(PersonaPath+'/' + Filename, 0))
        image = cv.imread(PersonaPath + '/' + Filename, 0)

    label = label + 1

face_Clasificacion = cv.face.LBPHFaceRecognizer_create()

print('Training....')
face_Clasificacion.train(faceData, np.array(labels))
face_Clasificacion.write('Model_ready.xml')
print('data saved successfully')

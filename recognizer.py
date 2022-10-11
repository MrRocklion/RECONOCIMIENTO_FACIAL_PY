import cv2 as cv
import os
import imutils as imu


User2 = 'Diaz2'
dataPath = "C:/Users/David/Documents/UNIVERSIDAD/PYTHON/PROYECTO_RECONOCIMIENTO_FACIAL/Data"
PersonaPath = dataPath + '/' + User2
if not os.path.exists(PersonaPath):
    print('Carpeta creada', PersonaPath)
    os.makedirs(PersonaPath)

capture = cv.VideoCapture(0)

face_Clasificacion = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

contador = 300

while True:

    ret, frame = capture.read()
    if ret == False: break

    frame = imu.resize(frame, width=640)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    auxframe = frame.copy()

    Detection = face_Clasificacion.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in Detection:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        Rostro = auxframe[y:y+h, x:x+w]
        Rostro = cv.resize(Rostro, (150, 150), interpolation=cv.INTER_CUBIC)
        cv.imwrite(PersonaPath + '/Frame_{}.png'.format(contador),Rostro)
        contador = contador+1

    cv.imshow('Face detection', frame)

    K = cv.waitKey(1)
    if K == ord('s') or contador >= 600:
        break

capture.release()
cv.destroyAllWindows()





import cv2 as cv
import os

dataPath = "C:/Users/David/Documents/UNIVERSIDAD/PYTHON/PROYECTO_RECONOCIMIENTO_FACIAL/Data"
imagenPaths = os.listdir(dataPath)
face_Clasificacion = cv.face.LBPHFaceRecognizer_create()
face_Clasificacion.read('Model_ready.xml')
capture = cv.VideoCapture(0)

face_Reconocedor = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = capture.read()
    if ret ==False: break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    auxframe = gray.copy()
    Rostros = face_Reconocedor.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in Rostros:

        Rostros = auxframe[y:y +h, x:x + w]
        Rostros = cv.resize(Rostros, (150,150), interpolation=cv.INTER_CUBIC)
        result = face_Clasificacion.predict(Rostros)

        cv.putText(frame, '{}'.format(result),(x, y-5), 1, 1.3, (255, 255, 0), 1, cv.LINE_AA)

        if result[1] < 70:
            cv.putText(frame, '{}'.format(imagenPaths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        else:
            cv.putText(frame, "Desconocido", (x, y-20), 2, 0.8, (0, 0, 255), 1, cv.LINE_AA)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow('Recognizer....', frame)

    K = cv.waitKey(1)
    if K == ord('f'):
        break
capture.release()
cv.destroyAllWindows()

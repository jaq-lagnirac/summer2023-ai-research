import os
import sys
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open camera")
    sys.exit(1)
                      
# face_path = os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_default.xml')
# eye_path = os.path.join(cv.data.haarcascades, 'haarcascade_eye.xml')

face_path = os.path.join('opencv-models', 'haarcascade_frontalface_default.xml')
eye_path = os.path.join('opencv-models', 'haarcascade_eye.xml')

face_cascade = cv.CascadeClassifier(face_path)
eye_cascade = cv.CascadeClassifier(eye_path)

# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

while True:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
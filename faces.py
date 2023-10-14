import pickle

import numpy as no
import cv2

# create haar cascade variable
face_cascade = cv2.CascadeClassifier("cascades\data\haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("config/trainer.yaml")

labels = []
with open("labels.pickle", "rb") as file:
    og_labels = pickle.load(file)
    print(og_labels)

# access webcam using opencv
cap = cv2.VideoCapture(0)
# print(labels)
while True:
    # capture frame by fra,e
    ret, frame = cap.read()
    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect the faces in the frame using haar cascade - face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in faces:
        # print(x,y,w,h)
        # get region of interest
        roi_gray = gray[y : y + h, x : x + w]
        roi_colour = frame[y : y + h, x : x + w]

        # create a recogniser
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_)
            print(og_labels[id_])
            font = cv2.FONT_HERSHEY_COMPLEX
            name = og_labels[id_]
            # print(type(name))
            colour = (0, 0, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, colour, stroke, cv2.LINE_AA)

        img_item = "face-image.png"
        cv2.imwrite(img_item, roi_gray)

        colour = (255, 0, 0)  # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), colour, stroke)

    # display the resulting frame
    cv2.imshow("frame", frame)
    # if no interaction in 20s and q button is pressed
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break
# release capture
cap.release()
cv2.destroyAllWindows()

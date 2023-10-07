import numpy as no
import cv2
# create haar cascade variable
face_cascade = cv2.CascadeClassifier("cascades\data\haarcascade_frontalface_alt2.xml")


# access webcam using opencv
cap = cv2.VideoCapture(0)

while True:
    # capture frame by fra,e
    ret, frame = cap.read()
    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect the faces in the frame using haar cascade - face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        # get region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_colour = frame[y:y+h, x:x+w]

        # create a recogniser

        
        img_item = "face-image.png"
        cv2.imwrite(img_item, roi_gray)

        colour = (255,0,0) # BGR 0-255
        stroke =2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), colour, stroke)



    # display the resulting frame
    cv2.imshow("frame", frame)
    # if no interaction in 20s and q button is pressed
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break
# release capture
cap.release()
cv2.destroyAllWindows()
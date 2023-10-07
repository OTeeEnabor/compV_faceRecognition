"""
This file will create the embeddings of a particular human face.
Embeddings are made with face_recognition.face_encodings methiod
"""
import sys
import cv2
import face_recognition
import pickle

# to identify the person in a pickle file, create name and a unique id for input
name = input("Enter name: \n")
ref_id = input("Enter id: \n")

# create a pickle file and dictionary to store face encodings
try:
    file = open("ref_name.pkl", "rb")
    ref_dict = pickle.load(file)
    file.close()
except:
    ref_dict = {}
# set the value for the key id as name
ref_dict[ref_id] = name

file = open("ref_name.pkl", "wb")
pickle.dump(ref_dict, file)
file.close()

try:
    file = open("ref_embed.pkl","rb")

    embed_dict = pickle.load(file)
    file.close()
except:
    embed_dict = {}

# create functionality to interact with webcam and create embeddings

# Single "triplet" training 


for i in range(5):
    key = cv2.waitKey(1)
    # initialise web cam
    webcam = cv2.VideoCapture(0)

    while True:
        check, frame = webcam.read()

        cv2.imshow("Capturing", frame)
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:,:,::-1]

        key = cv2.waitKey(1)

        if key == ord('s'):
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if face_locations !=[]:
                face_encoding = face_recognition.face_encodings(frame)[0]
                if ref_id in embed_dict:
                    embed_dict[ref_id] += [face_encoding]
                else:
                    embed_dict[ref_id] = [face_encoding]
                webcam.release()
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                break
        elif key == ord('q'):
            print("Done. Turning the camera off now.")
            webcam.release()
            print("camera off")
            print("program ended")
            cv2.destroyAllWindows()
            break


file=open("ref_embed.pkl","wb")
pickle.dump(embed_dict,file)
file.close()
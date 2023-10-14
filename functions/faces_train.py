import os
import pickle

import cv2
import numpy as np
import pandas as pd
from PIL import Image

FACE_CASCADE = cv2.CascadeClassifier("cascades\data\haarcascade_frontalface_alt2.xml")

RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()

def list_unique(new_list):
    unique_list = []
    for item in new_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list

def get_current_file(current_file):
    base_dir = os.path.dirname(current_file)
    return base_dir


def get_image_labels(base_dir):
    image_dir = os.path.join(base_dir, "images")
    current_id = 0
    label_ids = {}
    id_list = []
    label_list = []
    path_list = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("JPG") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path))
                label = label.replace(" ", "-").lower()
                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
            id_list.append(id_)
            label_list.append(label)
            path_list.append(path)
    label_dict = {"id": id_list, "label": label_list, "path": path_list}
    label_df = pd.DataFrame(label_dict)
    return label_df


def image_to_numpy_array(img_df):
    path_list = img_df["path"].tolist()
    id_list = img_df["id"].tolist()
    labels = img_df["label"].tolist()
    y_labels = []
    x_train = []
    for i in range(len(path_list)):
        # create pil image
        pil_image = Image.open(path_list[i]).convert("L")  # grayscale
        img_array = np.array(pil_image, "uint8")  # 8 bit image
        # detect faces
        faces = FACE_CASCADE.detectMultiScale(img_array)

        for x, y, w, h in faces:
            # define region of interest
            roi = img_array[y : y + h, x : x + w]
            # add region of interest to training list
            x_train.append(roi)
            y_labels.append(id_list[i])
    # create a dictionary with labels and ids
    #{"0": "label"}
    id_unique = list_unique(id_list)
    label_unique = list_unique(labels)
    label_dict = {id_unique[i]:label_unique[i] for i in range(len(id_unique))}
    with open("labels.pickle", "wb") as file:
        pickle.dump(label_dict, file)
    return x_train, y_labels


def face_recognizer(x_train, y_labels):
    RECOGNIZER.train(x_train, np.array(y_labels))
    RECOGNIZER.save("trainer.yaml")

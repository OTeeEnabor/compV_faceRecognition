import cv2
import face_recognition

from functions import faces_train

# training - this can be combined into one function
# get current base directory
base_directory = faces_train.get_current_file(__file__)
# get image ids and img_paths
img_ids,_,img_paths = faces_train.get_image_labels(base_directory)
# create x_train as np and labels using image ids and paths
x_train, y_labels = faces_train.image_to_numpy_array(img_ids, img_paths)
# train face recogniser using x_train, and labels
faces_train.face_recognizer(x_train, y_labels)


import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# from keras import backend as K
# from tensorflow import Graph, Session

# defining face detector
labels_dict={'drought': 0,
            'earthquake': 1,
            'flooding': 2,
            'normal': 3,
            'storm': 4,
            'tsunami': 5,
            'volcano': 6,
            'whirlwind': 7}
labels_list = ['Drought', 'Earthquake', 'Flooding', 'Normal', 'Storm', 'Tsunami', 'Volcano', 'Whirlwind']

color_dict={0:(0,255,0),1:(0,0,255)}
# global loaded_model
# graph1 = Graph()
# with graph1.as_default():
# 	session1 = Session(graph=graph1)
# 	with session1.as_default():
# 		loaded_model = pickle.load(open('Combined_Model.p', 'rb'))
model = load_model('./../model-best.h5')
class VideoCamera(object):
    def __init__(self, file_name=None):
        # capturing video
        self.video = cv2.VideoCapture(file_name if file_name else 0)
    def __del__(self):
        # releasing camera
        self.video.release()
    def get_frame(self):
        # extracting frames
        (rval, im) = self.video.read()
        # getting image dimensions
        im = cv2.resize(im, (224, 224))
        im = np.asarray(im)
        im = im / 255
        im = np.expand_dims(im, axis=0)
        classe_num = np.argmax(model.predict(im))
        label = labels_list[classe_num]

        cv2.putText(im, label, (10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', im)
        return jpeg.tobytes()
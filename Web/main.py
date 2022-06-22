import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import cv2
import numpy as np

from camera import VideoCamera
model = load_model('./model_best_acc_0.795918345451355.h5')
labels_list = ['Drought', 'Earthquake', 'Flooding', 'Normal', 'Storm', 'Tsunami', 'Volcano', 'Whirlwind']

def process_input(image):
    image = cv2.resize(image, (224, 224))
    image = np.asarray(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    classe_num = np.argmax(model.predict(image))
    label = labels_list[classe_num]
    return label

def predict_img(img):
    label = process_input(img)
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    # img = cv2.imencode('.jpg', img)[1].tobytes()
    return img

@app.route('/')
def upload_form():
	return render_template('upload.html')

def gen(filename):
    cap = cv2.VideoCapture(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # save video to static/uploads/
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    out = cv2.VideoWriter(os.path.join(app.config['UPLOAD_FOLDER'], 'output.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = predict_img(frame)
        out.write(frame)
        if i == length:
            break
        i += 1
        print(i, ' / ', length)
        (flag, encodedImage) = cv2.imencode(".jpg", frame)

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +  bytearray(encodedImage) + b'\r\n')

@app.route('/', methods=['POST'])
def upload_video():
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            print('upload_video filename: ' + filename)
            return Response(gen(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/display/<filename>')
def display_video(filename):
	#print('display_video filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()

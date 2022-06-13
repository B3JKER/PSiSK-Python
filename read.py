import cv2
import keras
from PIL import Image
import numpy as np
import datetime
import pyrebase
import time

loops = 1
model = keras.models.load_model('HeartAttack5Epochs.h5', compile=True)
BLUE = [255, 0, 0]

config = {
    "apiKey": "AIzaSyDE70DY6z_j8NEo3ls3VO_tbHaDA8e0jXQ",
    "authDomain": "patient-simulator.firebaseapp.com",
    "databaseURL": "https://patient-simulator-default-rtdb.europe-west1.firebasedatabase.app",
    "projectId": "patient-simulator",
    "storageBucket": "patient-simulator.appspot.com",
    "messagingSenderId": "1069997834909",
    "appId": "1:1069997834909:web:aad47310df2ef0ffb7ca74",
    "measurementId": "G-QN55VW7KV2",
}

firebase = pyrebase.initialize_app(config)
database = firebase.database()

while True:
    INPUT_SIZE = 500
    dataset = []
    ECG = []
    ECG_index = -1
    ECG_min = 0
    ECG_max = 0
    image = cv2.imread('blank.png')

    data = database.child('statuses').child("0").get()
    data = data.val()
    for i in data:
        ECG.append(data[i]['ECG'])
        ECG_index += 1
        if ECG[ECG_index] < ECG_min:
            ECG_min = ECG[ECG_index]
        if ECG[ECG_index] > ECG_max:
            ECG_max = ECG[ECG_index]

    ECG_index = 0
    previous_x = 0
    ECG = ECG[::-1]
    ECG = ECG[:527]
    ECG = ECG[::-1]
    for x in ECG:
        ECG_index += 2
        if ECG_index < 1053:
            image[int(113 - (x - ECG_min) / (ECG_max - ECG_min) * (113 - 61)), 97 + ECG_index] = BLUE
            image[int(113 - (x - ECG_min) / (ECG_max - ECG_min) * (113 - 61)), 98 + ECG_index] = BLUE
            cv2.line(image, [98 + ECG_index, int(113 - (x - ECG_min) / (ECG_max - ECG_min) * (113 - 61))],
                     [98 + ECG_index, int(113 - (previous_x - ECG_min) / (ECG_max - ECG_min) * (113 - 61))], BLUE, 1)
            previous_x = x

    # file_name = 'data' + str(loops) + '.png'
    # cv2.imwrite(file_name, image)

    image = Image.fromarray(image, 'RGB')
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    dataset.append(np.array(image))
    dataset = np.array(dataset)

    predictions = model.predict(dataset)
    print(predictions)
    ts = datetime.datetime.now().timestamp() * 1000
    ts = int(ts)

    if predictions[0] == 0:
        database.child('diagnoses').child("0").set({ts: 'zdrowy'})
    if predictions[0] == 1:
        database.child('diagnoses').child("0").set({ts: 'atak serca'})
    loops += 1
    time.sleep(0.9)

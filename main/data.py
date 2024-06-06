import cv2
#import csv
import numpy as np
import time
import face_recognition
import mediapipe as mp
import pytesseract
import matplotlib.pyplot as plt
import os
from inference_sdk import InferenceHTTPClient
from rppg.predict_vitals_torch import predict_vitals
from functions import *

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="CSj5C8ux5AKYMP231vm9")

# Inisialisasi variabel untuk teks
hc_txt = ['','','']
text_results = ['','','','','','','','','','','','','']
face_locations = []
face_encodings = []
face_landmarks_list=[]
face_names = []
known_faces = []
names = ['', '', '']
cards = ['', '', '']
names1 = ['', '', ''] #nambah
cards1 = ['', '', ''] #nambah
current_name=[]#nambah
fold_state=False #nambah
processed_photos = []
process_this_frame = True
bluff_flag = False 
first_player=0
x=(1920/1365)
y=(1080/767)
# Tentukan daftar area pemotongan dan pengolahan
photo_areas = [(501*y, 574*y, 68*x, 132*x), (604*y, 677*y, 68*x, 132*x), (400*y, 473*y, 68*x, 132*x)]

# Definisikan daftar label beserta koordinatnya
labels = [
   ((round(133*x), round(603*y)), (round(235*x), round(651*y)), (0, 255, 0)),   # handcard 2
    ((round(1011*x), round(613*y)), (round(1298*x), round(672*y)), (0, 255, 255)),  # communitycard 2
    ((round(133*x), round(653*y)), (round(285*x), round(678*y)), (0, 0, 255)),  # name 2
    ((round(133*x), round(680*y)), (round(322*x), round(704*y)), (255, 0, 200)),  # status 2
    ((round(275*x), round(620*y)), (round(324*x), round(650*y)), (100, 0, 255)),  # probability 2
    ((round(65*x), round(680*y)), (round(133*x), round(705*y)), (255, 0, 255)),  # deposit 2
    ((round(1170*x), round(675*y)), (round(1298*x), round(701*y)), (255, 255, 255)),  # pot
    ((round(133*x), round(550*y)), (round(285*x), round(575*y)), (0, 0, 255)), #1
    ((round(133*x), round(500*y)), (round(235*x), round(548*y)), (0, 255, 0)),
    ((round(133*x), round(577*y)), (round(322*x), round(600*y)), (255, 0, 200)),
    ((round(275*x), round(520*y)), (round(324*x), round(550*y)), (100, 0, 255)),
    ((round(65*x), round(575*y)), (round(133*x), round(600*y)), (255, 0, 255)), #1
    ((round(133*x), round(400*y)), (round(235*x), round(448*y)), (0, 255, 0)), #3
    ((round(133*x), round(447*y)), (round(285*x), round(472*y)), (0, 0, 255)),
    ((round(133*x), round(474*y)), (round(322*x), round(498*y)), (255, 0, 200)),
    ((round(275*x), round(420*y)), (round(324*x), round(450*y)), (100, 0, 255)),
    ((round(65*x), round(475*y)), (round(133*x), round(500*y)), (255, 0, 255)) #3
]

# Daftar area potongan yang akan diproses
areas = [
    (500*y, 600*y, 65*x, 323*x),   # handcard1
    (603*y, 704*y, 65*x, 323*x),   # handcard2
    (400*y, 497*y, 65*x, 323*x),   # handcard3
    (613*y, 701*y, 1011*x, 1298*x),# communitycard
    (550*y, 575*y, 133*x, 278*x),  # name1
    (653*y, 678*y, 133*x, 278*x),  # name2
    (447*y, 472*y, 133*x, 278*x),  # name3
    (577*y, 600*y, 133*x, 322*x),  # status1
    (680*y, 704*y, 133*x, 322*x),  # status2
    (474*y, 498*y, 133*x, 322*x),  # status3
    (578*y, 600*y, 68*x, 133*x),   # deposit1
    (681*y, 700*y, 68*x, 133*x),   # deposit2
    (476*y, 496*y, 68*x, 133*x),   # deposit3
    (520*y, 551*y, 274*x, 323*x),  # prob1
    (621*y, 650*y, 274*x, 323*x),  # prob2
    (421*y, 445*y, 274*x, 323*x),  # prob3
    (675*y, 701*y, 1170*x, 1298*x)   # pot
]

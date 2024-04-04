import cv2
import time
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from inference_sdk import InferenceHTTPClient

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="CSj5C8ux5AKYMP231vm9")

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Fungsi praproses
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def read_handcard(**result):
    text=''
    #print(time.perf_counter())
    for prediction in result['predictions']:
        if 78.0 <= prediction['x'] <= 84.5 and prediction['confidence'] > 0.6:
            #print(prediction['class'], end='')
            #Dict["0"]=prediction['class']
            text+=prediction['class']
    for prediction in result['predictions']:
        if 100.5 <= prediction['x'] <= 105.5 and prediction['confidence'] > 0.6:
            #print(prediction['class'], end=' ')
            #Dict["1"]=prediction['class']
            text+=prediction['class']
            text+=' '
    for prediction in result['predictions']:
        if 128.5 <= prediction['x'] <= 135.5 and prediction['confidence'] > 0.6:
            #print(prediction['class'], end='')
            #Dict["2"]=prediction['class']
            text+=prediction['class']
    for prediction in result['predictions']:
        if 151.0 <= prediction['x'] <= 156.0 and prediction['confidence'] > 0.6:
            #print(prediction['class'])
            #Dict["3"]=prediction['class']
            text+=prediction['class']
    #for i in range(num):
    #    text+=Dict[i]
    #    if(i==1):
    #        text+=" "
    text=text.upper()
    #print(time.perf_counter())
    #exit()
    return text

def read_comcard(**result):
  text=''
  for prediction in result['predictions']:
    if 11.0 <= prediction['x'] <= 18.5 and prediction['confidence'] > 0.6:
        #print(prediction['class'], end='')
        #Dict["0"]=prediction['class']
        text+=prediction['class']
  for prediction in result['predictions']:
    if 36.5 <= prediction['x'] <= 42.5 and prediction['confidence'] > 0.6:
        #print(prediction['class'], end=' ')
        #Dict["1"]=prediction['class']
        text+=prediction['class']
        text+=' '
  for prediction in result['predictions']:
    if 70.5 <= prediction['x'] <= 73.0 and prediction['confidence'] > 0.6:
        #print(prediction['class'], end='')
        #Dict["2"]=prediction['class']
        text+=prediction['class']
  for prediction in result['predictions']:
    if 95.5 <= prediction['x'] <= 98.5 and prediction['confidence'] > 0.6:
        #print(prediction['class'], end=' ')
        #Dict["3"]=prediction['class']
        text+=prediction['class']
        text+=' '
  for prediction in result['predictions']:
    if 127.5 <= prediction['x'] <= 131.0 and prediction['confidence'] > 0.6:
        #print(prediction['class'], end='')
        #Dict["4"]=prediction['class']
        text+=prediction['class']
  for prediction in result['predictions']:
    if 153.5 <= prediction['x'] <= 157.5 and prediction['confidence'] > 0.6:
        #print(prediction['class'], end=' ')
        #Dict["5"]=prediction['class']
        text+=prediction['class']
        text+=' '
  for prediction in result['predictions']:
    if 184.5 <= prediction['x'] <= 189.5 and prediction['confidence'] > 0.6:
        #print(prediction['class'], end='')
        #Dict["6"]=prediction['class']
        text+=prediction['class']
  for prediction in result['predictions']:
    if 209.0 <= prediction['x'] <= 214.5 and prediction['confidence'] > 0.6:
        #print(prediction['class'], end=' ')
        #Dict["7"]=prediction['class']
        text+=prediction['class']
        text+=' '
  for prediction in result['predictions']:
    if 241.5 <= prediction['x'] <= 246.0 and prediction['confidence'] > 0.6:
        #print(prediction['class'], end='')
        #Dict["8"]=prediction['class']
        text+=prediction['class']
  for prediction in result['predictions']:
    if 265.0 <= prediction['x'] <= 270.5 and prediction['confidence'] > 0.6:
        #print(prediction['class'])
        #Dict["9"]=prediction['class']
        text+=prediction['class']
  #for i in range(num):
  #  text+=Dict[i]
  #  if(i==1 or i==3 or i==5 or i==7):
  #      text+=" "
  text=text.upper()
  return text

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(150, 150))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

# Buka video
cap = cv2.VideoCapture(r'C:\Users\WILLIAM\Downloads\poker27.avi')  # Ganti 'video.mp4' dengan nama video Anda

frame_width = 1365
frame_height = 767
frame_size = (frame_width,frame_height)

output = cv2.VideoWriter('output_video_from_file.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Tampilkan frame yang telah dimodifikasi dengan mempertahankan proporsi aspek
    aspect_ratio = frame.shape[1] / frame.shape[0]  # Proporsi aspek bingkai
    new_width = 1365  # Lebar yang diinginkan
    new_height = int(new_width/aspect_ratio)# Tinggi yang sesuai dengan proporsi aspek
    frame = cv2.resize(frame, (new_width, new_height))  # Mengubah ukuran bingkai
    
    #labelling
    cv2.rectangle(frame, (133, 603), (235,651 ), (0, 255, 0)) #handcard
    cv2.rectangle(frame, (1011, 613), (1298,672 ), (0, 255, 255)) #communitycard
    cv2.rectangle(frame, (133, 653), (285,678 ), (0, 0, 255)) #name
    cv2.rectangle(frame, (133, 680), (322,704 ), (255, 0, 200)) #status
    cv2.rectangle(frame, (275, 620), (324,650 ), (100, 0, 255)) #probability 2
    cv2.rectangle(frame, (65, 680), (133,705 ), (255, 0, 255)) #deposit 2
    cv2.rectangle(frame, (1170, 675), (1298,701 ), (255, 255, 255)) #pot
    
    
    # Mendefinisikan area potongan gambar
    handcard1 = frame[500:600, 65:323]
    handcard2 = frame[603:704, 65:323]
    handcard3 = frame[400:497, 65:323]
    name1 = frame[550:575, 133:278]
    name2 = frame[653:678, 133:278]
    name3 = frame[447:472, 133:278]
    status1 = frame[577:600, 133:322]
    status2 = frame[680:704, 133:322]
    status3 = frame[474:498, 133:322]
    deposit1 = frame[578:600, 68:133]
    deposit2 = frame[681:700, 68:133]
    deposit3 = frame[476:496, 68:133]
    prob1 = frame[520:551, 274:323]
    prob2 = frame[621:650, 274:323]
    prob3 = frame[421:445, 274:323]
    communitycard = frame[613:701, 1011:1298]
    pot = frame[675:701, 1170:1298]

    # Praproses gambar
    name1_gray = get_grayscale(name1)
    name1_thresh = thresholding(name1_gray)

    name2_gray = get_grayscale(name2)
    name2_thresh = thresholding(name2_gray)

    name3_gray = get_grayscale(name3)
    name3_thresh = thresholding(name3_gray)

    status1_gray = get_grayscale(status1)
    status1_thresh = thresholding(status1_gray)

    status2_gray = get_grayscale(status2)
    status2_thresh = thresholding(status2_gray)

    status3_gray = get_grayscale(status3)
    status3_thresh = thresholding(status3_gray)

    pot_gray = get_grayscale(pot)
    pot_thresh = thresholding(pot_gray)
    
    deposit1_gray = get_grayscale(deposit1)
    deposit1_thresh = thresholding(deposit1_gray)

    deposit2_gray = get_grayscale(deposit2)
    deposit2_thresh = thresholding(deposit2_gray)

    deposit3_gray = get_grayscale(deposit3)
    deposit3_thresh = thresholding(deposit3_gray)

    prob1_gray = get_grayscale(prob1)
    prob1_thresh = thresholding(prob1_gray)

    prob2_gray = get_grayscale(prob2)
    prob2_thresh = thresholding(prob2_gray)

    prob3_gray = get_grayscale(prob3)
    prob3_thresh = thresholding(prob3_gray)

    # Ekstrak teks menggunakan Tesseract
    name1_txt=pytesseract.image_to_string(name1_thresh, config='--oem 3 --psm 6')
    name2_txt=pytesseract.image_to_string(name2_thresh, config='--oem 3 --psm 6')
    name3_txt=pytesseract.image_to_string(name3_thresh, config='--oem 3 --psm 6')
    status1_txt=pytesseract.image_to_string(status1_thresh, config='--oem 3 --psm 6')
    status2_txt=pytesseract.image_to_string(status2_thresh, config='--oem 3 --psm 6')
    status3_txt=pytesseract.image_to_string(status3_thresh, config='--oem 3 --psm 6')
    deposit1_txt=pytesseract.image_to_string(deposit1_thresh, config='--oem 3 --psm 6')
    deposit2_txt=pytesseract.image_to_string(deposit2_thresh, config='--oem 3 --psm 6')
    deposit3_txt=pytesseract.image_to_string(deposit3_thresh, config='--oem 3 --psm 6')
    prob1_txt=pytesseract.image_to_string(prob1_thresh, config='--oem 3 --psm 6')
    prob2_txt=pytesseract.image_to_string(prob2_thresh, config='--oem 3 --psm 6')
    prob3_txt=pytesseract.image_to_string(prob3_thresh, config='--oem 3 --psm 6')
    pot_txt=pytesseract.image_to_string(pot, config='--oem 3 --psm 6')
    #print(time.perf_counter())
    handcard1_txt = CLIENT.infer(handcard1, model_id="stage-2-cards/10")
    handcard2_txt = CLIENT.infer(handcard2, model_id="stage-2-cards/10")
    handcard3_txt = CLIENT.infer(handcard3, model_id="stage-2-cards/10")
    communitycard_txt = CLIENT.infer(communitycard, model_id="stage-2-cards/10")
    #print(time.perf_counter())
    #exit()
    hc1_txt=read_handcard(**handcard1_txt)
    hc2_txt=read_handcard(**handcard2_txt)
    hc3_txt=read_handcard(**handcard3_txt)
    cc_txt=read_comcard(**communitycard_txt)
    
    faces = detect_bounding_box(
        frame
    )

    # Tambahkan teks ke frame
    if(len(hc3_txt)>3):
        cv2.rectangle(frame, (133, 400), (235,448 ), (0, 255, 0)) #handcard
        cv2.rectangle(frame, (133, 447), (285,472 ), (0, 0, 255)) #name
        cv2.rectangle(frame, (133, 474), (322,498 ), (255, 0, 200)) #status
        cv2.rectangle(frame, (275, 420), (324,450 ), (100, 0, 255)) #probability 3
        cv2.rectangle(frame, (65, 475), (133,500 ), (255, 0, 255)) #deposit 3
        frame=cv2.putText(frame, name3_txt[:len(name3_txt)-1], (335, 467), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        if(len(status3_txt)>3):
            frame=cv2.putText(frame, status3_txt[:len(status3_txt)-1], (335, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 200), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, hc3_txt, (335, 427), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, deposit3_txt[:len(deposit3_txt)-2], (1, 488), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, prob3_txt[:len(prob3_txt)-2], (335, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 0, 255), 2, cv2.LINE_AA)

    if(len(hc1_txt)>3):
        cv2.rectangle(frame, (133, 550), (285,575 ), (0, 0, 255)) #name
        cv2.rectangle(frame, (133, 500), (235,548 ), (0, 255, 0)) #handcard
        cv2.rectangle(frame, (133, 577), (322,600 ), (255, 0, 200)) #status
        cv2.rectangle(frame, (275, 520), (324,550 ), (100, 0, 255)) #probability 1
        cv2.rectangle(frame, (65, 575), (133,600 ), (255, 0, 255)) #deposit 1
        frame=cv2.putText(frame, name1_txt[:len(name1_txt)-1], (335, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        if(len(status1_txt)>3):
            frame=cv2.putText(frame, status1_txt[:len(status1_txt)-1], (335, 595), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 200), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, hc1_txt, (335, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, deposit1_txt[:len(deposit1_txt)-2], (1, 595), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, prob1_txt[:len(prob1_txt)-2], (335, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 0, 255), 2, cv2.LINE_AA)

    frame=cv2.putText(frame, name2_txt[:len(name2_txt)-1], (335, 673), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    if(len(status2_txt)>3):
        frame=cv2.putText(frame, status2_txt[:len(status2_txt)-1], (335, 699), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 200), 2, cv2.LINE_AA)
    frame=cv2.putText(frame, hc2_txt, (335, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    frame=cv2.putText(frame, deposit2_txt[:len(deposit2_txt)-2], (1, 699), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2, cv2.LINE_AA)
    frame=cv2.putText(frame, prob2_txt[:len(prob2_txt)-2], (335, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 0, 255), 2, cv2.LINE_AA)

    frame=cv2.putText(frame, pot_txt[:len(pot_txt)-1], (1190, 730), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    frame=cv2.putText(frame, cc_txt, (1011, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Frame', frame)
    
    # Tambahkan logika untuk keluar saat menekan tombol 'q' atau menutup jendela
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan sumber daya
cap.release()
output.release()
cv2.destroyAllWindows()
print("sucsess")
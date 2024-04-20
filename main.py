import cv2
import time
import face_recognition
import pytesseract
import numpy as np
from inference_sdk import InferenceHTTPClient

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="CSj5C8ux5AKYMP231vm9")

# Fungsi praproses
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def preprocess(img, mode='RGB'):
    if mode:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img)

def read_handcard(**result):
    text=''
    #print(time.perf_counter())
    for prediction in result['predictions']:
        if 78.0 <= prediction['x'] <= 84.5 and prediction['confidence'] > 0.6:
            #Dict["0"]=prediction['class']
            text+=prediction['class']
    for prediction in result['predictions']:
        if 100.5 <= prediction['x'] <= 105.5 and prediction['confidence'] > 0.6:
            #Dict["1"]=prediction['class']
            text+=prediction['class']
            text+=' '
    for prediction in result['predictions']:
        if 128.5 <= prediction['x'] <= 135.5 and prediction['confidence'] > 0.6:
            #Dict["2"]=prediction['class']
            text+=prediction['class']
    for prediction in result['predictions']:
        if 151.0 <= prediction['x'] <= 156.0 and prediction['confidence'] > 0.6:
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
    if 13.0 <= prediction['x'] <= 18.5 and prediction['confidence'] > 0.6:
        #Dict["0"]=prediction['class']
        text+=prediction['class']
  for prediction in result['predictions']:
    if 36.5 <= prediction['x'] <= 42.5 and prediction['confidence'] > 0.6:
        #Dict["1"]=prediction['class']
        text+=prediction['class']
        text+=' '
  for prediction in result['predictions']:
    if 70.5 <= prediction['x'] <= 73.0 and prediction['confidence'] > 0.6:
        #Dict["2"]=prediction['class']
        text+=prediction['class']
  for prediction in result['predictions']:
    if 95.5 <= prediction['x'] <= 98.5 and prediction['confidence'] > 0.6:
        #Dict["3"]=prediction['class']
        text+=prediction['class']
        text+=' '
  for prediction in result['predictions']:
    if 127.5 <= prediction['x'] <= 131.0 and prediction['confidence'] > 0.6:
        #Dict["4"]=prediction['class']
        text+=prediction['class']
  for prediction in result['predictions']:
    if 153.5 <= prediction['x'] <= 157.5 and prediction['confidence'] > 0.6:
        #Dict["5"]=prediction['class']
        text+=prediction['class']
        text+=' '
  for prediction in result['predictions']:
    if 184.5 <= prediction['x'] <= 189.5 and prediction['confidence'] > 0.6:
        #Dict["6"]=prediction['class']
        text+=prediction['class']
  for prediction in result['predictions']:
    if 209.0 <= prediction['x'] <= 214.5 and prediction['confidence'] > 0.6:
        #Dict["7"]=prediction['class']
        text+=prediction['class']
        text+=' '
  for prediction in result['predictions']:
    if 241.5 <= prediction['x'] <= 246.0 and prediction['confidence'] > 0.6:
        #Dict["8"]=prediction['class']
        text+=prediction['class']
  for prediction in result['predictions']:
    if 265.0 <= prediction['x'] <= 270.5 and prediction['confidence'] > 0.6:
        #Dict["9"]=prediction['class']
        text+=prediction['class']
  #for i in range(num):
  #  text+=Dict[i]
  #  if(i==1 or i==3 or i==5 or i==7):
  #      text+=" "
  text=text.upper()
  return text

hc1_txt=''
hc2_txt=''
hc3_txt=''
face_locations = []
face_encodings = []
face_names = []
known_faces = []
names=['','','']
process_this_frame=True
path=r"C:\Users\WILLIAM\Downloads\poker19.avi"
# Buka video
cap = cv2.VideoCapture(path)  # Ganti 'video.mp4' dengan nama video Anda
ret, frame=cap.read()
aspect_ratio = frame.shape[1] / frame.shape[0]  # Proporsi aspek bingkai
new_width = 1365  # Lebar yang diinginkan
new_height = int(new_width/aspect_ratio)# Tinggi yang sesuai dengan proporsi aspek
frame = cv2.resize(frame, (new_width, new_height))
photo1=frame[501:574, 68:132]
photo2=frame[604:677, 68:132]
photo3=frame[400:473, 68:132]
photo1=preprocess(photo1)
photo2=preprocess(photo2)
photo3=preprocess(photo3)

if face_recognition.face_locations(photo1):
    face1_face_encoding = face_recognition.face_encodings(photo1)[0]
    known_faces.append(face1_face_encoding)

face2_face_encoding = face_recognition.face_encodings(photo2)[0]
known_faces.append(face2_face_encoding)

if face_recognition.face_locations(photo3):
    face3_face_encoding = face_recognition.face_encodings(photo3)[0]
    known_faces.append(face3_face_encoding)

cap.release()

cap = cv2.VideoCapture(path)
frame_width = 1365
frame_height = 767
frame_size = (frame_width,frame_height)
output = cv2.VideoWriter('output_video_from_file.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, frame_size)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame=-1
face_flag = False

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    current_frame+=1
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
    print(f"Processing frame {current_frame} of {total_frames}")
    # Ekstrak teks menggunakan Tesseract
    if current_frame % 30==0:
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
    if current_frame % 32==0:
        handcard1_txt = CLIENT.infer(handcard1, model_id="stage-2-cards/10")
        hc1_txt=read_handcard(**handcard1_txt)
    if current_frame % 33==0:
        handcard2_txt = CLIENT.infer(handcard2, model_id="stage-2-cards/10")
        hc2_txt=read_handcard(**handcard2_txt)
    if current_frame % 34==0:
        handcard3_txt = CLIENT.infer(handcard3, model_id="stage-2-cards/10")
        hc3_txt=read_handcard(**handcard3_txt)
    if current_frame % 35==0:
        communitycard_txt = CLIENT.infer(communitycard, model_id="stage-2-cards/10")
        cc_txt=read_comcard(**communitycard_txt)
    
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

    if(names[0]==''):
        names[0]=name1_txt
    if(names[1]==''):
        names[1]=name2_txt
    if(names[2]==''):
        names[2]=name3_txt
    
    if process_this_frame:
        face_flag=False
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame,1)
        for (top, right, bottom, left) in face_locations:
            if right-left>=100:
                face_locations=face_locations
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []
        #print(face_locations)
        for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)

                # If you had more than 2 faces, you could make this logic a lot prettier
                # but I kept it simple for the demo
            name = None
            card = None
            if match[0]:
                name = names[0][:len(names[0])-1]
                card = hc1_txt
                face_flag=True
            if len(known_faces)>=2:
                if match[1]:
                    name = names[1][:len(names[1])-1]
                    card = hc2_txt
                    face_flag=True
            if len(known_faces)==3:
                if match[2]:
                    name = names[2][:len(names[2])-1]
                    card = hc3_txt
                    face_flag=True

            face_names.append(name)
    #process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        if not face_flag:
            continue
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, card, (right - 56, bottom - 6), font, 0.5, (255, 255, 255), 1)

    output.write(frame)
    cv2.imshow('Frame', frame)
    
    # Tambahkan logika untuk keluar saat menekan tombol 'q' atau menutup jendela
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan sumber daya
cap.release()
output.release()
cv2.destroyAllWindows()
print("sucsess")
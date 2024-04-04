import cv2
import time
import face_recognition
import pytesseract
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Fungsi praproses
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def preprocess(img, mode='RGB'):
    if mode:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img)

face_locations = []
face_encodings = []
face_names = []
process_this_frame=True
path=r"C:\Users\WILLIAM\Downloads\poker26.avi"

cap = cv2.VideoCapture(path)  # Ganti 'video.mp4' dengan nama video Anda
ret, frame=cap.read()
aspect_ratio = frame.shape[1] / frame.shape[0]  # Proporsi aspek bingkai
new_width = 1365  # Lebar yang diinginkan
new_height = int(new_width/aspect_ratio)# Tinggi yang sesuai dengan proporsi aspek
frame = cv2.resize(frame, (new_width, new_height))
known_faces = []
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
output = cv2.VideoWriter('output_video_face_recog.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)


while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Tampilkan frame yang telah dimodifikasi dengan mempertahankan proporsi aspek
    aspect_ratio = frame.shape[1] / frame.shape[0]  # Proporsi aspek bingkai
    new_width = 1365  # Lebar yang diinginkan
    new_height = int(new_width/aspect_ratio)# Tinggi yang sesuai dengan proporsi aspek
    frame = cv2.resize(frame, (new_width, new_height))  # Mengubah ukuran bingkai
    
    name1 = frame[550:575, 133:278]
    name2 = frame[653:678, 133:278]
    name3 = frame[447:472, 133:278]
    
    name1_gray = get_grayscale(name1)
    name1_thresh = thresholding(name1_gray)
    name2_gray = get_grayscale(name2)
    name2_thresh = thresholding(name2_gray)
    name3_gray = get_grayscale(name3)
    name3_thresh = thresholding(name3_gray)

    name1_txt=pytesseract.image_to_string(name1_thresh, config='--oem 3 --psm 6')
    name2_txt=pytesseract.image_to_string(name2_thresh, config='--oem 3 --psm 6')
    name3_txt=pytesseract.image_to_string(name3_thresh, config='--oem 3 --psm 6')

    frame=cv2.putText(frame, name1_txt[:len(name1_txt)-1], (335, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    frame=cv2.putText(frame, name2_txt[:len(name2_txt)-1], (335, 673), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame,1)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)

            name = None
            if match[0]:
                name = name1_txt[:len(name1_txt)-1]
            elif match[1]:
                name = name2_txt[:len(name2_txt)-1]

            face_names.append(name)

    process_this_frame = not process_this_frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        if not name:
            continue
        if right-left>=100:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

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
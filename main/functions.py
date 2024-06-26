import cv2
import numpy as np

x=(1920/1365)

# Fungsi praproses
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def preprocess(img, mode='RGB'):
    if mode:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img)

def read_handcard(**result):
    text = ''
    for prediction in result['predictions']:
        if 78.0*x <= prediction['x'] <= 84.5*x and prediction['confidence'] > 0.6:
            text += prediction['class']
    for prediction in result['predictions']:
        if 100.5*x <= prediction['x'] <= 105.5*x and prediction['confidence'] > 0.6:
            text += prediction['class']
            text += ' '
    for prediction in result['predictions']:
        if 128.5*x <= prediction['x'] <= 135.5*x and prediction['confidence'] > 0.6:
            text += prediction['class']
    for prediction in result['predictions']:
        if 151.0*x <= prediction['x'] <= 156.0*x and prediction['confidence'] > 0.6:
            text += prediction['class']
    text = text.upper()
    return text

def read_comcard(**result):
    text = ''
    for prediction in result['predictions']:
        if 13.0*x <= prediction['x'] <= 18.5*x and prediction['confidence'] > 0.6:
            text += prediction['class']
    for prediction in result['predictions']:
        if 36.5*x <= prediction['x'] <= 42.5*x and prediction['confidence'] > 0.6:
            text += prediction['class']
            text += ' '
    for prediction in result['predictions']:
        if 69.5*x <= prediction['x'] <= 74.0*x and prediction['confidence'] > 0.6:
            text += prediction['class']
    for prediction in result['predictions']:
        if 95.5*x <= prediction['x'] <= 98.5*x and prediction['confidence'] > 0.6:
            text += prediction['class']
            text += ' '
    for prediction in result['predictions']:
        if 127.5*x <= prediction['x'] <= 131.0*x and prediction['confidence'] > 0.6:
            text += prediction['class']
    for prediction in result['predictions']:
        if 153.5*x-1 <= prediction['x'] <= 157.5*x+1 and prediction['confidence'] > 0.6:
            text += prediction['class']
            text += ' '
    for prediction in result['predictions']:
        if 184.5*x <= prediction['x'] <= 189.5*x and prediction['confidence'] > 0.6:
            text += prediction['class']
    for prediction in result['predictions']:
        if 209.0*x <= prediction['x'] <= 214.5*x and prediction['confidence'] > 0.6:
            text += prediction['class']
            text += ' '
    for prediction in result['predictions']:
        if 241.5*x <= prediction['x'] <= 246.0*x and prediction['confidence'] > 0.6:
            text += prediction['class']
    for prediction in result['predictions']:
        if 265.0*x <= prediction['x'] <= 270.5*x and prediction['confidence'] > 0.6:
            text += prediction['class']
    text = text.upper()
    return text

# Fungsi untuk menggambar kotak pada frame
def draw_box(frame, left, top, right, bottom, name, card):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, card, (right - 56, bottom - 6), font, 0.5, (255, 255, 255), 1)
    
def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka video {video_path}")
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        duration = frame_count / fps
    else:
        print("Error: Tidak dapat mendapatkan frame rate")
        return None
    cap.release()
    
    return duration

def draw_face_landmarks(face_landmarks, frame):
    color = (0, 0, 255)
    thickness = 1
    landmarks_parts = {
        'nose_bridge': False,
        'chin': False,
        'left_eyebrow': False,
        'right_eyebrow': False,
        'nose_tip': False,
        'left_eye': True,
        'right_eye': True,
        'top_lip': True,
        'bottom_lip': True
    }
    for part, is_closed in landmarks_parts.items():
        points = np.array(face_landmarks[part], dtype=np.int32)*4
        cv2.polylines(frame, [points], is_closed, color, thickness)
    for part in face_landmarks:
        for (x, y) in face_landmarks[part]:
            cv2.circle(frame, (x*4, y*4), 1, color, -1)


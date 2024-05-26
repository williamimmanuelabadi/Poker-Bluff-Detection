from data import *

def sample_photos(frame):
    # Iterasi melalui daftar area dan memproses setiap gambar
    for area in photo_areas:
        photo = frame[round(area[0]):round(area[1]), round(area[2]):round(area[3])]
        processed_photo = preprocess(photo)
        processed_photos.append(processed_photo)

    # Daftar foto yang akan diproses
    for photo in processed_photos:
        face_locations = face_recognition.face_locations(photo)
        if face_locations:
            face_encoding = face_recognition.face_encodings(photo)[0]
            known_faces.append(face_encoding)
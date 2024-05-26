from data import *
import playersphoto

# Video path
path = r"C:\Users\WILLIAM\Downloads\poker27.avi"

# Process photo of players
cap = cv2.VideoCapture(path)  
ret, frame = cap.read()
new_width = 1920
new_height = 1080 
frame_size = (new_width, new_height)
# frame = cv2.resize(frame, frame_size)
playersphoto.sample_photos(frame)
cap.release()

# Start video processing
cap = cv2.VideoCapture(path)
# output = cv2.VideoWriter('output_video_from_file.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (300,370))
output = cv2.VideoWriter('output_video_from_file.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, frame_size)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = -1

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("Sucessfully read")
        break
    current_frame += 1
    # frame = cv2.resize(frame, frame_size)

    areas_list = []
    preprocessed_images = []
    for i, (top, bottom, left, right) in enumerate(areas):
        area = frame[round(top):round(bottom), round(left):round(right)]
        areas_list.append(area)
        if i > 3:
            gray = get_grayscale(area)
            thresh = thresholding(gray)
            preprocessed_images.append(thresh)

    if current_frame % 30==0:
        for i, image in enumerate(preprocessed_images):
            text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
            text_results[i]=text
    if current_frame % 32==0:
        handcard1_txt = CLIENT.infer(areas_list[0], model_id="stage-2-cards/10")
        hc_txt[0]=read_handcard(**handcard1_txt)
    if current_frame % 33==0:
        handcard2_txt = CLIENT.infer(areas_list[1], model_id="stage-2-cards/10")
        hc_txt[1]=read_handcard(**handcard2_txt)
    if current_frame % 34==0:
        handcard3_txt = CLIENT.infer(areas_list[2], model_id="stage-2-cards/10")
        hc_txt[2]=read_handcard(**handcard3_txt)
    if current_frame % 35==0:
        communitycard_txt = CLIENT.infer(areas_list[3], model_id="stage-2-cards/10")
        cc_txt=read_comcard(**communitycard_txt)
    
    for i in range(3):
        if len(cards[i]) != 5:
            cards[i] = hc_txt[i]
        if names[i] == '':
            names[i] = text_results[i]
 
    if(len(hc_txt[0])>3):
        for label in labels[7:12]:
            start_point, end_point, color = label
            cv2.rectangle(frame, start_point, end_point, color)
        frame=cv2.putText(frame, text_results[0][:len(text_results[0])-1], (round(335*x), round(570*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        if(len(text_results[3])>3):
            frame=cv2.putText(frame, text_results[3][:len(text_results[3])-1], (round(335*x), round(595*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 200), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, hc_txt[0], (round(335*x), round(530*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, text_results[6][:len(text_results[6])-2], (round(1*x), round(595*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, text_results[9][:len(text_results[9])-2], (round(335*x), round(550*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 0, 255), 2, cv2.LINE_AA)
    if(len(hc_txt[1])>3):
        for label in labels[:7]:
            start_point, end_point, color = label
            cv2.rectangle(frame, start_point, end_point, color)
        frame=cv2.putText(frame, text_results[1][:len(text_results[1])-1], (round(335*x), round(673*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        if(len(text_results[4])>3):
            frame=cv2.putText(frame, text_results[4][:len(text_results[4])-1], (round(335*x), round(699*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 200), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, hc_txt[1], (round(335*x), round(630*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, text_results[7][:len(text_results[7])-2], (round(1*x), round(699*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, text_results[10][:len(text_results[10])-2], (round(335*x), round(650*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 0, 255), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, text_results[12][:len(text_results[12])-1], (round(1190*x), round(730*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, cc_txt, (round(1011*x), round(600*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    if(len(hc_txt[2])>3):
        for label in labels[12:]:
            start_point, end_point, color = label
            cv2.rectangle(frame, start_point, end_point, color)
        frame=cv2.putText(frame, text_results[2][:len(text_results[2])-1], (round(335*x), round(467*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        if(len(text_results[5])>3):
            frame=cv2.putText(frame, text_results[5][:len(text_results[5])-1], (round(335*x), round(483*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 200), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, hc_txt[2], (round(335*x), round(427*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, text_results[8][:len(text_results[8])-2], (round(1*x), round(488*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, text_results[11][:len(text_results[11])-2], (round(335*x), round(450*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 0, 255), 2, cv2.LINE_AA)

    
    mp_face_detection = mp.solutions.face_detection
    if process_this_frame:
        face_flag=False
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = small_frame[:, :, ::-1]
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
            results = face_detection.process(rgb_frame)
        face_locations = []
        face_landmarks_list=[]
        if results.detections:
            for detection in results.detections:
                top = int(detection.location_data.relative_bounding_box.ymin*100*2*y)
                right = int((detection.location_data.relative_bounding_box.xmin+detection.location_data.relative_bounding_box.width)*100*3.5*x)
                bottom = int((detection.location_data.relative_bounding_box.ymin+detection.location_data.relative_bounding_box.height)*100*2*y)
                left = int(detection.location_data.relative_bounding_box.xmin*100*3.5*x)
                face_locations.append((top, right, bottom, left))
            for (top, right, bottom, left) in face_locations:
                if right-left>=40:
                    face_landmarks_list = face_recognition.face_landmarks(rgb_frame,[(top, right, bottom, left)])
                    face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding,tolerance=0.6)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            name = None
            card = None
            if matches[best_match_index]:
                name = names[best_match_index][:len(names[best_match_index])-1]
                card = cards[best_match_index]
                face_flag=True
    process_this_frame = not process_this_frame
    for face_landmarks in face_landmarks_list:
        top=round(min(face_landmarks['nose_bridge'][2][1]*4-230*y,face_landmarks['left_eyebrow'][4][1]*4-175*y))
        right=round(max(face_landmarks['nose_bridge'][2][0]*4+150*x,face_landmarks['chin'][16][0]*4+45*x))
        bottom=round(max(face_landmarks['nose_bridge'][2][1]*4+140*y,face_landmarks['chin'][8][1]*4+45*y))
        left=round(min(face_landmarks['nose_bridge'][2][0]*4-150*x,face_landmarks['chin'][0][0]*4-45*x))
        if not face_flag:
            continue
        draw_box(frame, left, top, right, bottom, name, card)
    
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 1080, 1920)
    cv2.imshow('Frame', frame)
    output.write(frame)
    print(f"finish frame {current_frame} of {total_frames}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
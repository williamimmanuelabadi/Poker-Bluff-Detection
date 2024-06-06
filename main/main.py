from data import *
import playersphoto

video_number=27

# Video path
path = fr"C:\Users\WILLIAM\Downloads\poker{video_number}.avi"

# Process photo of players
cap = cv2.VideoCapture(path)  
ret, frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
playersphoto.sample_photos(frame)
cap.release()

# initialize variables
top1=0
right1=0
left1=0
bottom1=0
current_face=''
round_num = 1
folder=f'D:\Code\Project poker\Poker{video_number}'

# name of output video
def get_video_filename(round_num):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return os.path.join(folder, f'face{round_num}.avi')
output_name = get_video_filename(round_num)

# Start video processing
cap = cv2.VideoCapture(path)
output = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (1000,1000))
full_output = cv2.VideoWriter(os.path.join(folder, f'full_output.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (1920,1080))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = -1
a=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("Sucessfully read")
        break
    current_frame += 1
    current_player=0
    fold=-1
    min_prob=100
    min_prob_pos=-1
    aa=0

    # time of video
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_seconds = total_frames / fps
    current_seconds = current_frame / fps
    minutes = int(current_seconds // 60)
    seconds = int(current_seconds % 60)

    # draw rectangle
    areas_list = []
    preprocessed_images = []
    for i, (top, bottom, left, right) in enumerate(areas):
        area = frame[round(top):round(bottom), round(left):round(right)]
        areas_list.append(area)
        if i > 3:
            gray = get_grayscale(area)
            thresh = thresholding(gray)
            preprocessed_images.append(thresh)

    # OCR
    if current_frame % fps==0:
        for i, image in enumerate(preprocessed_images):
            text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
            text_results[i]=text
        print("Time: {:02d}:{:02d} ---------------------------------------------".format(minutes, seconds))
        bluff_flag=False

    # card recognition
    if current_frame % (fps+2)==0:
        handcard1_txt = CLIENT.infer(areas_list[0], model_id="stage-2-cards/10")
        hc_txt[0]=read_handcard(**handcard1_txt)
    if current_frame % (fps+3)==0:
        handcard2_txt = CLIENT.infer(areas_list[1], model_id="stage-2-cards/10")
        hc_txt[1]=read_handcard(**handcard2_txt)
    if current_frame % (fps+4)==0:
        handcard3_txt = CLIENT.infer(areas_list[2], model_id="stage-2-cards/10")
        hc_txt[2]=read_handcard(**handcard3_txt)
    if current_frame % (fps+5)==0:
        communitycard_txt = CLIENT.infer(areas_list[3], model_id="stage-2-cards/10")
        cc_txt=read_comcard(**communitycard_txt)
    
    # save the name and card(avoid name or card detection not accurate)
    for i in range(3):
        if len(cards[i]) != 5:
            cards[i] = hc_txt[i]
            cards1[i] = hc_txt[i]
        if names[i] == '':
            names[i] = text_results[i]
            names1[i] = text_results[i]
 
    # put the text 
    # middle player
    if(len(hc_txt[0])>3):
        current_player+=1
        for label in labels[7:12]:
            start_point, end_point, color = label
            cv2.rectangle(frame, start_point, end_point, color)
        frame=cv2.putText(frame, text_results[0][:len(text_results[0])-1], (round(335*x), round(570*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        if(len(text_results[3])>3):
            frame=cv2.putText(frame, text_results[3][:len(text_results[3])-1], (round(335*x), round(595*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 200), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, hc_txt[0], (round(335*x), round(530*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, text_results[6][:len(text_results[6])-2], (round(1*x), round(595*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, text_results[9][:len(text_results[9])-2], (round(335*x), round(550*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 0, 255), 2, cv2.LINE_AA)
    
    # below player
    if(len(hc_txt[1])>3):
        current_player+=1
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
    
    # upper player
    if(len(hc_txt[2])>3):
        current_player+=1
        for label in labels[12:]:
            start_point, end_point, color = label
            cv2.rectangle(frame, start_point, end_point, color)
        frame=cv2.putText(frame, text_results[2][:len(text_results[2])-1], (round(335*x), round(467*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        if(len(text_results[5])>3):
            frame=cv2.putText(frame, text_results[5][:len(text_results[5])-1], (round(335*x), round(483*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 200), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, hc_txt[2], (round(335*x), round(427*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, text_results[8][:len(text_results[8])-2], (round(1*x), round(488*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2, cv2.LINE_AA)
        frame=cv2.putText(frame, text_results[11][:len(text_results[11])-2], (round(335*x), round(450*y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 0, 255), 2, cv2.LINE_AA)

    # fold player detction
    if(first_player == 0):
        first_player = current_player
    elif first_player > current_player and fold_state==False:
        fold_frame=current_frame
        fold_state=True
    elif fold_state and current_frame >= fold_frame+fps:
        for i in range(0,first_player):
            for j in range(0,current_player):
                if cards1[i] == hc_txt[j]:
                    continue
            fold=i
            break
        print(f"Fold player is {names1[fold].strip()} ")
        cards1[:]=hc_txt
        names1[:]=text_results[0:3]
        fold_state=False
        first_player = current_player

    # bluffing detection
    for i in range(9,9+current_player,1):
        if text_results[i][0:-2].isdigit():
            temp=int(text_results[i][:len(text_results[i])-2])
            if min_prob>temp:
                min_prob=temp
                min_prob_pos=i 
    if "BET" in text_results[min_prob_pos-6][:3] or "ALL" in text_results[min_prob_pos-6][:3] or "RAISE" in text_results[min_prob_pos-6][:5]:
        if bluff_flag==False:
            print(f"{names1[min_prob_pos-9].strip()} is bluffing ")
            bluff_flag=True
    
    # face detection
    mp_face_detection = mp.solutions.face_detection
    if process_this_frame:
        face_flag=False
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = small_frame[:, :, ::-1]
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3) as face_detection:
            results = face_detection.process(rgb_frame)
        face_locations = []
        face_landmarks_list=[]
        name = []
        card = []
        if results.detections:
            for detection in results.detections:
                top = int(detection.location_data.relative_bounding_box.ymin*100*2*y)
                right = int((detection.location_data.relative_bounding_box.xmin+detection.location_data.relative_bounding_box.width)*100*3.5*x)
                bottom = int((detection.location_data.relative_bounding_box.ymin+detection.location_data.relative_bounding_box.height)*100*2*y)
                left = int(detection.location_data.relative_bounding_box.xmin*100*3.5*x)
                if right-left>=50:
                    face_locations.append((top, right, bottom, left))
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame,face_locations)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # face recognition
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding,tolerance=0.6)
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name.append(names[best_match_index][:len(names[best_match_index])-1])
                card.append(cards[best_match_index])
                face_flag=True
    process_this_frame = not process_this_frame
    
    # for first face
    if face_flag:
        if(a!=1):
            if len(name[0])>0:
                a=1
            current_face=name[0]

    # face rectangle
    for face_landmarks, name1, card1 in zip(face_landmarks_list,name,card):
        if aa==0:
            top=round(face_landmarks['nose_bridge'][2][1]*4-500)
            right=round(face_landmarks['nose_bridge'][2][0]*4+500)
            bottom=round(face_landmarks['nose_bridge'][2][1]*4+500)
            left=round(face_landmarks['nose_bridge'][2][0]*4-500)
            if not face_flag:
                continue
            if(top<0):
                bottom-=top
                top=0
            if(left<0):
                right-=left
                left=0
            if(right<0):
                left+=right
                right=0
            top1=top
            right1=right
            bottom1=bottom
            left1=left
            draw_face_landmarks(face_landmarks, frame)
            draw_box(frame, left1, top1, right1, bottom1, name1, card1)
            aa+=1
    new_frame=frame[top1:bottom1, left1:right1]

    # facial changing detection
    if(len(current_name)==0):
        current_name.extend(name)
        if len(name)!=0:
            print("Facial changing ", end='')
            print(name)
    else:
        if(len(current_name)<len(name)):
            print("Facial changing ", end='')
            print(name)
        else:
            for i in current_name:
                if i not in name:
                    print("Facial changing ", end='')
                    print(name)
        current_name.clear()
        current_name.extend(name) 

    # make heart rate graph dan make new output file
    if face_flag:
        if((current_face != name[0] and len(name[0]) > 2)or current_frame>=(total_frames-2)):
            current_face=name[0]    
            output.release()
            if get_video_length(output_name)>=4.5:
                heart_rate=predict_vitals(output_name)
                y1=np.arange(0, 101, 0.5)
                y1 = y1[:len(heart_rate)]
                plt.figure()
                plt.plot(y1, heart_rate)
                plt.scatter(y1, heart_rate, color='red')
                plt.title(f'face{round_num}')
                plt.ylabel('Heart Rate (Bpm)')
                plt.xlabel('Time (s)')
                output_name = os.path.join(folder, f'face{round_num}.png')
                plt.savefig(output_name)
                plt.close()
            else:
                print(f'face{round_num} duration is smaller than 4.5 seconds')
            print(f'face{round_num} is finished')
            round_num += 1
            output_name = get_video_filename(round_num)
            output = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc('M','J','P','G'),fps, (1000,1000))
    
    # cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Frame', 1080, 1920)
    # cv2.imshow('Frame', frame)
    output.write(new_frame)
    full_output.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
full_output.release()
cv2.destroyAllWindows()

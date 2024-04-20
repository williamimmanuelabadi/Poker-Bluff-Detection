import cv2
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="CSj5C8ux5AKYMP231vm9")

def read_comcard1(**result):
    first=False
    for prediction in result['predictions']:
        if 11.0 <= prediction['x'] <= 18.5 and prediction['confidence'] > 0.6:
            first=True
    for prediction in result['predictions']:
        if 36.5 <= prediction['x'] <= 42.5 and prediction['confidence'] > 0.6:
            first=True
    return first

def read_comcard3(**result):
    third=False
    for prediction in result['predictions']:
        if 127.5 <= prediction['x'] <= 131.0 and prediction['confidence'] > 0.6:
            third=True
    for prediction in result['predictions']:
        if 153.5 <= prediction['x'] <= 157.5 and prediction['confidence'] > 0.6:
            third=True
    return third

def get_video_filename(round_num):
    return f'poker{round_num}.avi'

# Ganti path dengan lokasi video MP4 Anda
input_video_path = r'C:\Users\WILLIAM\Downloads\poker.mp4'

cap = cv2.VideoCapture(input_video_path)  

frame_width = 1365
frame_height = 767
frame_size = (frame_width,frame_height)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0

round_num = 1
stage_counter = 0

output_name = get_video_filename(round_num)
output = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc('M','J','P','G'), 30, frame_size)

flag = False

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    
    current_frame += 1
    print(f"Processing frame {current_frame} of {total_frames}")

    aspect_ratio = frame.shape[1] / frame.shape[0]  # Proporsi aspek bingkai
    new_width = 1365  # Lebar yang diinginkan
    new_height = int(new_width/aspect_ratio)  # Tinggi yang sesuai dengan proporsi aspek
    frame = cv2.resize(frame, (new_width, new_height))  # Mengubah ukuran bingkai

    communitycard = frame[613:701, 1011:1298]
    communitycard_txt = CLIENT.infer(communitycard, model_id="stage-2-cards/10")
    cc1_txt = read_comcard1(**communitycard_txt)
    cc3_txt = read_comcard3(**communitycard_txt)

    if not cc1_txt and flag:
        output.release()
        round_num += 1
        output_name = get_video_filename(round_num)
        output = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)
        flag = False
    elif cc3_txt:
        output.write(frame)
        print("Trim")
        flag = True

    #cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()
print("Success")

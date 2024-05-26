import glob
import os
import pandas as pd
import cv2
import h5py
import numpy as np
import scipy.io
import mediapipe as mp
#from mediapipe.tasks.python.vision import FaceAligner
import matplotlib.pyplot as plt
import random
import math
import cv2 as cv
from PIL import Image,ImageDraw
from typing_extensions import final

def get_nframe_video(path):
    temp_f1 = h5py.File(path, 'r')
    temp_dysub = np.array(temp_f1["dysub"])
    nframe_per_video = temp_dysub.shape[0]
    return nframe_per_video


def get_nframe_video_val(path):
    temp_f1 = scipy.io.loadmat(path)
    temp_dXsub = np.array(temp_f1["dXsub"])
    nframe_per_video = temp_dXsub.shape[0]
    return nframe_per_video


def split_subj(data_dir, cv_split, subNum):
    f3 = h5py.File(data_dir + '/M.mat', 'r')
    M = np.transpose(np.array(f3["M"])).astype(np.bool)
    subTrain = subNum[~M[:, cv_split]].tolist()
    subTest = subNum[M[:, cv_split]].tolist()
    return subTrain, subTest


def take_last_ele(ele):
    ele = ele.split('.')[0][-2:]
    try:
        return int(ele[-2:])
    except ValueError:
        return int(ele[-1:])


def sort_video_list(data_dir, taskList, subTrain):
    final = []
    for p in subTrain:
        for t in taskList:
            x = glob.glob(os.path.join(data_dir, 'P' + str(p) + 'T' + str(t) + 'VideoB2*.mat'))
            x = sorted(x)
            x = sorted(x, key=take_last_ele)
            final.append(x)
    return final


def RandomErasing(img,probability=0.1,sl=0.005,sh = 0.01,r1 = 0.3):
    if random.uniform(0, 1) > probability:
            return img
    for attempt in range(100):
        area = img.shape[0]* img.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)       #長寬比

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            if img.shape[2] == 3:
                img[ x1:x1 + h, y1:y1 + w,0] = 0
                img[ x1:x1 + h, y1:y1 + w,1] = 0
                img[ x1:x1 + h, y1:y1 + w,2] = 0
            else:
                img[ x1:x1 + h, y1:y1 + w,0] = 0
            return img

    return img

def read_video(videoFilePath,img_size = 72):
    vidObj = cv2.VideoCapture(videoFilePath)
    totalFrames = 0
    success, img = vidObj.read()
    while success:
        totalFrames += 1 # get total frame size
        success, img = vidObj.read()
        # cv2.imwrite('perspective_img/test_img.png',img)        #印圖片用
        # exit()
    vidObj = cv2.VideoCapture(videoFilePath)
    Xsub = np.zeros((totalFrames, img_size, img_size, 3), dtype = np.float32)       #(72,72)
    success, img = vidObj.read()
    i = 0
    while success:
        resized_img = cv2.resize(img, (img_size, img_size)) #nambah sendiri
        Xsub[i, :, :, :] = resized_img
        success, img = vidObj.read() # read the next one
        i = i + 1
    return Xsub

def image_random_rorate(Xsub,flip,rotate_pixel):
    '''
    Xsub = N*C*H*W*
    '''
    dim = Xsub.size(2)
    x1 = x2 = x3 = x4 = 0
    y1 = y2 = y3 = y4 = 0
    y = x_threshold = y_threshold = 0      #rorate parameter
    x = 5
    ##  random erase
    probability = 0.5   #erase 機率     default 0.1
    sl = 0.01           #最小比例
    sh = 0.05           #最大比例
    r1 = 0.3            #最小長寬比
    erase_flag = random.uniform(0, 1)
    if erase_flag > probability:
        erase = 0
    else:
        erase = 1
        area = dim* dim
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        if w < dim and dim:
            erase_x = random.randint(0, dim - h)
            erase_y = random.randint(0, dim - w)
    ##
    if random.uniform(0, 1) < 0.5:
        rorate = 0          #隨機轉
    else:
        rorate = 1          #順時針旋轉
    ##
    N_Xsub = np.zeros_like(Xsub).reshape(-1,dim,dim,3)
    for i in range(len(Xsub)):
        vidLxL = Xsub[i].reshape(dim,dim,3)
        if erase==1:
            #random mask
            vidLxL[ erase_x:erase_x + h, erase_y:erase_y + w,0] = 0
            vidLxL[ erase_x:erase_x + h, erase_y:erase_y + w,1] = 0
            vidLxL[ erase_x:erase_x + h, erase_y:erase_y + w,2] = 0
            #mask下半臉
            if erase_flag < 0.25:
                vidLxL[ int(dim/2):dim , : ,0] = 0
                vidLxL[ int(dim/2):dim , : ,1] = 0
                vidLxL[ int(dim/2):dim , : ,2] = 0
            else:   #上半臉
                vidLxL[ 0:int(dim/2) , : ,0] = 0
                vidLxL[ 0:int(dim/2) , : ,1] = 0
                vidLxL[ 0:int(dim/2) , : ,2] = 0
        vidLxL = cv2.cvtColor(np.asarray(vidLxL),cv2.COLOR_RGB2BGR)
        if rorate == 1:
            if x==5:
                x_threshold = 1
            if x==-5:
                x_threshold = 0
            
            if x_threshold==0:
                x+=1
            else:
                x-=1
                
            if y==5:
                y_threshold = 1
            if y==-5:
                y_threshold = 0
            
            if y_threshold==0:
                y+=1
            else:
                y-=1
            
            p1 = np.float32([[0,0],[dim,0],[0,dim],[dim,dim]])          #旋轉
            p2 = np.float32([[-x,-y],[dim-x,y],[x,dim-y],[dim-x,dim-y]])
            m = cv2.getPerspectiveTransform(p1,p2)
        else:
            x1 += np.random.choice([-1,0,1])  
            y1 += np.random.choice([-1,0,1]) 
            x2 += np.random.choice([-1,0,1])  
            y2 += np.random.choice([-1,0,1]) 
            x3 += np.random.choice([-1,0,1])  
            y3 += np.random.choice([-1,0,1]) 
            x4 += np.random.choice([-1,0,1])  
            y4 += np.random.choice([-1,0,1]) 
            x1 = rorate_func(x1,rotate_pixel)
            x2 = rorate_func(x2,rotate_pixel)
            x3 = rorate_func(x3,rotate_pixel)
            x4 = rorate_func(x4,rotate_pixel)
            y1 = rorate_func(y1,rotate_pixel)
            y2 = rorate_func(y2,rotate_pixel)
            y3 = rorate_func(y3,rotate_pixel)
            y4 = rorate_func(y4,rotate_pixel)
            p1 = np.float32([[0,0],[dim,0],[0,dim],[dim,dim]])          # 左上、右上、左下、右下
            p2 = np.float32([[x1,y1],[dim+x2,y2],[x3,dim+y3],[dim+x4,dim+y4]])
            m = cv2.getPerspectiveTransform(p1,p2)
        vidLxL = cv2.warpPerspective(vidLxL, m, (dim, dim))
        vidLxL = cv2.cvtColor(np.asarray(vidLxL),cv2.COLOR_BGR2RGB)
        
        if flip == 1: 
            N_Xsub[i, :, :, :] = cv2.flip(vidLxL, 1)
        else:
            N_Xsub[i, :, :, :] = vidLxL
        if (i+1) %180 == 0:
            # cv2.imwrite('perspective_img/test_img_rotate2.png',vidLxL)        #印圖片用
            # exit()
            x1 = x2 = x3 = x4 = y1 = y2 = y3 = y4 = y =  x_threshold = y_threshold = 0
            x = 5
    #
    # Xsub = N_Xsub.astype(np.uint8)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('/data/Users/huangkaichun/UBFC_pre/gt/TTTTEST.avi', fourcc, 30.0, (72,  72))
    # for frame_count in range(int(len(Xsub))):
    #     out.write(Xsub[frame_count])
    # out.release()
    # exit()
    #
    return N_Xsub.reshape(-1,3,dim,dim)

def rorate_func(input,rotate_pixel):
    
    if input>=rotate_pixel:
        input = rotate_pixel
    if input<=-rotate_pixel:
        input = -rotate_pixel
    return input

def preprocess_raw_video(videoFilePath, dim=72,augmentation = 0,num =0,file = ''):
    #########################################################################
    # set up
    if augmentation==1:
        r = 0   #rotate
        x = 0
        y = 0
        x_threshold = 0
        y_threshold = 0
        change = 0
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath)
    totalFrames = 0
    success, img = vidObj.read()
    while success:
        totalFrames += 1 # get total frame size
        success, img = vidObj.read()
        # cv2.imwrite('perspective_img/test_img.png',img)        #印圖片用
        # exit()
    vidObj = cv2.VideoCapture(videoFilePath)
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype = np.float32)
    Xsub_fusion = np.zeros((totalFrames, dim, dim, 3), dtype = np.float32)
    Xsub_mask = np.zeros((totalFrames, dim, dim, 3), dtype = np.float32)
    height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
    success, img = vidObj.read()

    
    # opticalAddimg = np.zeros((totalFrames, 480, 640, 3), dtype = np.float32)
    #########################################################################
    # Crop each frame size into dim x dim
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    while success:
        min_x = width
        min_y = height
        max_x = 0
        max_y = 0
        ldmks = np.zeros((468, 5), dtype=np.float32)
        ldmks[:, 0] = -1.0
        ldmks[:, 1] = -1.0
        with mp_face_mesh.FaceMesh(max_num_faces = 1,
                                    min_detection_confidence = 0.5,
                                    min_tracking_confidence = 0.5) as face_mash:
            results = face_mash.process(img)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [l for l in face_landmarks.landmark]
                # #33,263眼角對齊
                # landmark_33 = landmarks[33]
                # landmark_263 = landmarks[263]
                # landmark_58 = landmarks[58]
                # landmark_288 = landmarks[288]
                # # 計算相對角度
                # delta_x = landmark_263.x - landmark_33.x
                # delta_y = landmark_263.y - landmark_33.y
                # angle_radians = math.atan2(delta_y, delta_x)
                # delta_x = landmark_288.x - landmark_58.x
                # delta_y = landmark_288.y - landmark_58.y
                # angle_radians2 = math.atan2(delta_y, delta_x)
                # # 將弧度轉換為角度
                # angle = (int(math.degrees(angle_radians))+int(math.degrees(angle_radians2)))/2
                # # 旋轉
                # center = (int(landmark_33.x * width),int(landmark_33.y*height))
                # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                # # 使用旋轉矩陣進行仿射變換
                # img = cv2.warpAffine(img, rotation_matrix, (int(width), int(height)))
                # results = face_mash.process(img)
                # if results.multi_face_landmarks:
                #     face_landmarks = results.multi_face_landmarks[0]
                #     landmarks = [l for l in face_landmarks.landmark]         
                for idx in range(len(landmarks)):
                    landmark = landmarks[idx]
                    if not ((landmark.HasField('visibility') and landmark.visibility < 0.5)
                            or (landmark.HasField('presence') and landmark.presence < 0.5)):
                        coords = mp_drawing._normalized_to_pixel_coordinates(
                            landmark.x, landmark.y, width, height)
                        if coords:
                            ldmks[idx, 0] = coords[1]
                            ldmks[idx, 1] = coords[0]
                            if min_x>coords[0]:
                                min_x = coords[0]
                            if min_y>coords[1]: 
                                min_y = coords[1]
                            if max_x<coords[0]:
                                max_x = coords[0]
                            if max_y<coords[1]:
                                max_y = coords[1]
        #########################################################################################   mask
        # fusion_img ,mask_img= SkinExtractionConvexHull.extract_skin(img,ldmks)
        # fusion_img = cv2.resize(fusion_img, (dim, dim), interpolation = cv2.INTER_CUBIC)
        # mask_img = cv2.resize(mask_img, (dim, dim), interpolation = cv2.INTER_CUBIC)
        #######################################################################################
        img = img[min_y:max_y,min_x:max_x]
        vidLxL = cv2.resize(img, (dim, dim), interpolation = cv2.INTER_LINEAR)#cv2.INTER_CUBIC
        # cv2.imwrite('perspective_img/test_img{}.png'.format(i),vidLxL)        #印圖片用
        # exit()
        if augmentation==1:
            aug = 'aug'
            # vidLxL = RandomErasing(vidLxL)      #RandomErasing
            if change==0:
                if x==5:
                    x_threshold = 1
                if x==-5:
                    x_threshold = 0
                
                if x_threshold==0:
                    x+=1
                else:
                    x-=1
                if x==0:
                    change=1
            else:
                if y==5:
                    y_threshold = 1
                if y==-5:
                    y_threshold = 0
                
                if y_threshold==0:
                    y+=1
                else:
                    y-=1    
                if y==0:
                    change=0
            
            p1 = np.float32([[0,0],[dim,0],[0,dim],[dim,dim]])          #左右轉
            p2 = np.float32([[y,y],[dim-x,x],[y,dim-y],[dim-x,dim-x]])
            m = cv2.getPerspectiveTransform(p1,p2)
            vidLxL = cv2.warpPerspective(vidLxL, m, (dim, dim))
            # fusion_img = cv2.warpPerspective(fusion_img, m, (dim, dim))
            # mask_img = cv2.warpPerspective(mask_img, m, (dim, dim))
            # cv2.imwrite('perspective_img/{}perspective.png'.format(i),vidLxL)
            Xsub[i, :, :, :] = cv2.flip(vidLxL, 1)
            # Xsub_fusion[i, :, :, :] = cv2.flip(Xsub_fusion, 1)
            # Xsub_mask[i, :, :, :] = cv2.flip(Xsub_mask, 1)
        else:
            aug = 'I420_96'
            Xsub[i, :, :, :] = vidLxL
            # Xsub_fusion[i, :, :, :] = fusion_img
            # Xsub_mask[i, :, :, :] = mask_img
        
        success, img = vidObj.read() # read the next one
        i = i + 1
    ###################################################################################### optical flow
    Xsub = Xsub.astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*'I420')        #XVID
    out = cv2.VideoWriter('/data/Users/huangkaichun/UBFC_pre/{}/{}{}.avi'.format(file,num,aug), fourcc, 30.0, (dim,  dim))
    for frame_count in range(int(totalFrames)):
        out.write(Xsub[frame_count])
    out.release()
    
    # Xsub_fusion = Xsub_fusion.astype(np.uint8)
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # out = cv2.VideoWriter('/data/Users/huangkaichun/UBFC_pre/gt/aug_{}fusion.mp4'.format(num), fourcc, 30.0, (72,  72))
    # for frame_count in range(int(totalFrames)):
    #     out.write(Xsub_fusion[frame_count])
    # out.release()
    
    # Xsub_mask = Xsub_mask.astype(np.uint8)
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # out = cv2.VideoWriter('/data/Users/huangkaichun/UBFC_pre/gt/aug_{}mask.mp4'.format(num), fourcc, 30.0, (72,  72))
    # for frame_count in range(int(totalFrames)):
    #     out.write(Xsub_mask[frame_count])
    # out.release()
    ##########################################################################################
    return Xsub

def bbox2_CPU(img):
    """
    Args:
        img (ndarray): ndarray with shape [rows, columns, rgb_channels].

    Returns: 
        Four cropping coordinates (row, row, column, column) for removing black borders (RGB [O,O,O]) from img.
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    nzrows = np.nonzero(rows)
    nzcols = np.nonzero(cols)
    if nzrows[0].size == 0 or nzcols[0].size == 0:
        return -1, -1, -1, -1
    rmin, rmax = np.nonzero(rows)[0][[0, -1]]
    cmin, cmax = np.nonzero(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

class SkinExtractionConvexHull:
    def extract_skin(image, ldmks):
        """
        This method extract the skin from an image using Convex Hull segmentation.

        Args:
            image (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
            ldmks (float32 ndarray): landmarks used to create the Convex Hull; ldmks is a ndarray with shape [num_landmarks, xy_coordinates].

        Returns:
            Cropped skin-image and non-cropped skin-image; both are uint8 ndarray with shape [rows, columns, rgb_channels].
        """
        aviable_ldmks = ldmks[ldmks[:,0] >= 0][:,:2]            
        # face_mask convex hull 
        hull = scipy.spatial.ConvexHull(aviable_ldmks)
        verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
        img = Image.new('L', image.shape[:2], 0)
        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
        mask = np.array(img)
        mask = np.expand_dims(mask,axis=0).T
        # left eye convex hull
        left_eye_ldmks = ldmks[MagicLandmarks.left_eye]
        aviable_ldmks = left_eye_ldmks[left_eye_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = scipy.spatial.ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            left_eye_mask = np.array(img)
            left_eye_mask = np.expand_dims(left_eye_mask,axis=0).T
        else:
            left_eye_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)
        
        # right eye convex hull
        right_eye_ldmks = ldmks[MagicLandmarks.right_eye]
        aviable_ldmks = right_eye_ldmks[right_eye_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = scipy.spatial.ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            right_eye_mask = np.array(img)
            right_eye_mask = np.expand_dims(right_eye_mask,axis=0).T
        else:
            right_eye_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

        # mounth convex hull
        mounth_ldmks = ldmks[MagicLandmarks.mounth]
        aviable_ldmks = mounth_ldmks[mounth_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = scipy.spatial.ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            mounth_mask = np.array(img)
            mounth_mask = np.expand_dims(mounth_mask,axis=0).T
        else:
            mounth_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

        # apply masks and crop 
        
        skin_image = image * mask* (1-left_eye_mask) * (1-right_eye_mask) * (1-mounth_mask)     #mask全黑
        rmin, rmax, cmin, cmax = bbox2_CPU(skin_image)

        cropped_skin_im_mask = skin_image
        if rmin >= 0 and rmax >= 0 and cmin >= 0 and cmax >= 0 and rmax-rmin >= 0 and cmax-cmin >= 0:
            cropped_skin_im_mask = skin_image[int(rmin):int(rmax), int(cmin):int(cmax)]
            
        #blur
        skin_image = image * mask
        rmin, rmax, cmin, cmax = bbox2_CPU(image*left_eye_mask)
        cropped_skin_im_left_eye = skin_image[int(rmin):int(rmax), int(cmin):int(cmax)]
        cropped_skin_im_left_eye = cv2.GaussianBlur(cropped_skin_im_left_eye,(15,15),0)
        skin_image[int(rmin):int(rmax),int(cmin):int(cmax)] = cropped_skin_im_left_eye
        
        rmin, rmax, cmin, cmax = bbox2_CPU(image*right_eye_mask)
        cropped_skin_im_right_eye = skin_image[int(rmin):int(rmax), int(cmin):int(cmax)]
        cropped_skin_im_right_eye = cv2.GaussianBlur(cropped_skin_im_right_eye,(15,15),0)
        skin_image[int(rmin):int(rmax),int(cmin):int(cmax)] = cropped_skin_im_right_eye
        
        rmin, rmax, cmin, cmax = bbox2_CPU(image*mounth_mask)
        cropped_skin_im_mounth = skin_image[int(rmin):int(rmax), int(cmin):int(cmax)]
        cropped_skin_im_mounth = cv2.GaussianBlur(cropped_skin_im_mounth,(15,15),0)
        skin_image[int(rmin):int(rmax),int(cmin):int(cmax)] = cropped_skin_im_mounth
        
        rmin, rmax, cmin, cmax = bbox2_CPU(skin_image)
        cropped_skin_im = skin_image
        if rmin >= 0 and rmax >= 0 and cmin >= 0 and cmax >= 0 and rmax-rmin >= 0 and cmax-cmin >= 0:
            cropped_skin_im = skin_image[int(rmin):int(rmax), int(cmin):int(cmax)]
        #

        return cropped_skin_im, cropped_skin_im_mask
    
class MagicLandmarks():
    # left_eye = [105,63,70,139,34,227,123,50,36,142,198,174,122,193,55,66]
    # right_eye = [334,293,300,368,264,447,352,280,266,371,420,399,351,417,285,296]
    left_eye = [157,144, 145, 22, 23, 25, 154, 31, 160, 33, 46, 52, 53, 55, 56, 189, 190, 63, 65, 66, 70, 221, 222, 223, 225, 226, 228, 229, 230, 231, 232, 105, 233, 107, 243, 124]
    right_eye = [384, 385, 386, 259, 388, 261, 265, 398, 276, 282, 283, 285, 413, 293, 296, 300, 441, 442, 445, 446, 449, 451, 334, 463, 336, 464, 467, 339, 341, 342, 353, 381, 373, 249, 253, 255]
    mounth = [391, 393, 11, 269, 270, 271, 287, 164, 165, 37, 167, 40, 43, 181, 313, 314, 186, 57, 315, 61, 321, 73, 76, 335, 83, 85, 90, 106]




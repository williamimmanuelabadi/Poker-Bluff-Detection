import torch
import scipy.io
import pandas as pd
import numpy as np
import cv2
#import inference_preprocess
import scipy.io
from rppg.pre_process import preprocess_raw_video,read_video
from scipy.signal import butter,find_peaks


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, dataset_path, label,  signal='pulse', input_channel='raw', img_size=72,
                 num_gpus=0,window_size = 180,flag = 0,augmentation=0,test = 0,stride = 30):            #default window_size = 180
        self.dataset_path = dataset_path
        self.fs = stride        #SOTA 10    default 30
        self.signal = signal
        self.input_channel = input_channel
        self.img_size = img_size
        self.num_gpus = num_gpus
        #self.label = label #nambah
        self.window_size = window_size
        self.flag = flag
        self.augmentation = augmentation
        if self.input_channel =='raw':
            # self.output = inference_preprocess.preprocess_raw_video(self.dataset_path, dim=img_size,augmentation=self.augmentation)    #TSCAN
            self.output = read_video(self.dataset_path,img_size = img_size) 
            #self.label = self.data_load_func(label) #nambah
            if test == 0:
                self.training_len = int((len(self.output)-self.window_size*2)/self.fs)          #window_size*2 意思是減掉往後6步和最後6秒的validation
            else:    
                self.training_len = int((len(self.output)-self.window_size)/self.fs)               #給test用的，不會切validation出來
            self.validation_len = int(self.window_size/self.fs+1)

    def __len__(self):
        if self.flag == 0:
            return self.training_len
        else:
            return self.validation_len

    def data_load_func(self, path):
        data = pd.read_csv(path,header=None)
        data = np.array(data).reshape(-1)
        # [b_pulse, a_pulse] = butter(1, [0.6 / 30 * 2, 3 / 30 * 2], btype='bandpass')
        # data = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(data))
        return np.asarray(data)

    def __getitem__(self, index):
        if self.input_channel =='raw':
            if self.flag == 0:
                output = self.output[index*self.fs:index*self.fs+self.window_size]  
                #label = self.label[index*self.fs:index*self.fs+self.window_size] #nambah
            else:
                output = self.output[(index+self.training_len-1)*self.fs:(index+self.training_len-1)*self.fs+self.window_size]      
                label = self.label[(index+self.training_len-1)*self.fs:(index+self.training_len-1)*self.fs+self.window_size] 
        output = np.float32(output)
        #label = np.float32(label) #nambah
        # return (output, label)
        return (output)

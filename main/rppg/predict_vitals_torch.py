import math
import numpy as np
import pandas as pd
import scipy.io
import os
import sys
import argparse
import torch
sys.path.append('../')
import cv2
import matplotlib.pyplot as plt
from rppg.modelv1 import EfficientPhys_residualTSM,EfficientPhys_attention #,bpm_model
from torch.utils.data import DataLoader
from rppg.data_generator import DataGenerator
from scipy.signal import butter
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# from torchsummary import summary
X = [38,39,40,41,42,43,44,45,46,47,48,49]
# X = [11,12]
# X = [42,43,44,45,46,47,48,49]
# X = [41]
# X = [1,2,3,4]
ex = 'ex6'
exnum = '1-1'
# after-exercise
input_channel = 'raw'
test_data = 'UBFC'
def predict_vitals(path_data):
    for num in X:
        exnum = num
        if input_channel =='raw':
            img = 'vid.avi'
            img_size = 72
            stride = 15
            model_checkpoint = r'D:\Code\Project poker\main\rppg\EfficientPhys_model85.pt'
            # path_data = '/data/Users/huangkaichun/UBFC/after-exercise/{}'.format(img)
            # path_label = '/data/Users/huangkaichun/UBFC/Dataset_1/Cleaned/after-exercise.csv'
            
            path_data = path_data.format(num)    
            # path_label = '/data/Users/huangkaichun/UBFC/Dataset_1/Cleaned/{}BVP.csv'.format(num)
            # path_data = r'D:\Code\Project poker\Poker20\face1.avi'.format(num)
            # path_label = '/data/Users/huangkaichun/UBFC/Dataset_2/Cleaned/{}BVP_fixed.csv'.format(num)  #num
            # time_label = pd.read_csv('/data/Users/huangkaichun/UBFC/Dataset_2/Cleaned/{}BVP_time.csv'.format(num),header=None)
            # path_data = '/data/Users/huangkaichun/UBFC/other/25s_poker.mp4'
            # path_data = '/data/Users/huangkaichun/專題生data/{}/{}.mp4'.format(ex,exnum)
            cap = cv2.VideoCapture(path_data)
            totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fs = fps = float(cap.get(5))
            window_size = 120    #256 SOTA       default 180
            branch = float(1/fps*1000)
        
        training_dataset = []

        [b_pulse, a_pulse] = butter(1, [0.6 / fs * 2, 3 / fs * 2], btype='bandpass')

        training_dataset.append(DataGenerator(path_data, img_size=img_size, num_gpus=torch.cuda.device_count(),
                                            label= "",flag=0,window_size = window_size,input_channel = input_channel,test = 1,stride=stride))

        tr_dataloader = DataLoader(training_dataset[0], batch_size=1, shuffle=False, drop_last=True)


        label = []
        bpm_label = []
        bpm_predict = []
        predict = []
        with torch.no_grad():
            device = torch.device("cpu")
            model = EfficientPhys_attention(frame_depth = window_size,img_size=img_size,in_channels=3) #EfficientPhys_residualTSM   EfficientPhys_attention
            model.eval()
            # model = TSCAN(img_size=img_size,frame_depth = window_size)      #tscan
            model.load_state_dict(torch.load(model_checkpoint, map_location=torch.device('cpu')))
            model = model.to(device)
            for i, data in enumerate(tr_dataloader):
                # inputs, labels  = data[0].to(device), data[1].to(device) #nambah
                inputs= data[0].to(device)
                print(inputs.shape)
                # inputs = inputs.view(-1, 6, img_size, img_size)           #TSCAN
                if input_channel =='raw':
                    inputs = inputs.view(-1, 3, img_size, img_size)         #EFFI
                    #labels = labels.view(-1) #nambah
                    outputs = model(inputs)
                    outputs = outputs[:,0]
                    #label.append(labels.data.cpu().numpy()) #nambah
                    outputs = outputs.data.cpu().numpy()
                    predict.append(outputs)

        # if test_data=='UBFC':
            # label = np.array(label).reshape(-1)
            # label = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(label))
        pulse_pred = np.array(predict).reshape(-1)
        pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred)) #nambah padlen

        # pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))
        # plt.plot(np.abs(scipy.fft.rfft(pulse_pred,n = 1024)))
        # plt.savefig('img/{}fre.png'.format(num),dpi=300)
        # plt.clf()

    

        stride = float(window_size/fps)           #預測步長
        # stride = 10
        height = 0
        distance = 1

        predict_peaks, _ = find_peaks(pulse_pred, height=height, distance=distance)

        height = np.mean(pulse_pred[predict_peaks])-np.std(pulse_pred[predict_peaks])
        distance = 10

        predict_peaks, _ = find_peaks(pulse_pred, height=height, distance=distance)

        # plt.subplot(211)
        # plt.plot(label)       
        # plt.plot(true_peaks, label[true_peaks], "x")

        # plt.subplot(212)
        # plt.plot(pulse_pred)
        # plt.plot(predict_peaks, pulse_pred[predict_peaks], "x")
        # plt.show()
        
        T_medianbpm = []
        P_medianbpm = []
        # fps = round(fps)
        frame = float(len(pulse_pred)/fps/stride)

        fft_len = 16384
        for i in range(int(frame)):
            bvps = pulse_pred[int(i*fps*stride):int((i+1)*stride*fps)]    
            fft = np.abs(scipy.fft.rfft(bvps,n = fft_len))
            bpm_place = np.argmax(fft)
            bpm = scipy.fft.rfftfreq(fft_len,1/fps)
            P_medianbpm.append(bpm[bpm_place]*60)
            # if test_data=='UBFC':
            #     bvps = label[int(i*fps*stride):int((i+1)*stride*fps)]
            #     fft = np.abs(scipy.fft.rfft(bvps,n = fft_len))
            #     bpm_place = np.argmax(fft)
            #     bpm = scipy.fft.rfftfreq(fft_len,1/fps)
            #     T_medianbpm.append(bpm[bpm_place]*60)   
        # print(P_medianbpm)
        ################## sec stage #######################
        # bmodel = bpm_model()
        # bmodel.eval()
        # bmodel.load_state_dict(torch.load('model_checkpoint/BPM_model_miniloss.pt'))
        # bmodel = bmodel.to(device)
        # P_medianbpm = np.array(P_medianbpm)
        # input_bpm_len = len(P_medianbpm)
        # if input_bpm_len<200:
        #     P_medianbpm.resize(200)
        # P_medianbpm = torch.tensor(P_medianbpm,dtype=torch.float32).reshape(-1,200)
        # P_medianbpm = P_medianbpm.to(device)
        
        # P_medianbpm = bmodel(P_medianbpm)
        # P_medianbpm = P_medianbpm.data.cpu().numpy()
        # P_medianbpm = P_medianbpm.reshape(-1)
        # P_medianbpm = P_medianbpm[0:input_bpm_len]

        # pd.DataFrame(T_medianbpm).to_csv('img/{}Tbpm.csv'.format(num),index=False)
        # pd.DataFrame(P_medianbpm).to_csv('img/{}Pbpm.csv'.format(num),index=False)
        ########## Plot ##################

        # ssr_predict = pd.read_csv('2SRtest//{}bpm_data.csv'.format(num),header=None).transpose()
        
        # minum = len(T_medianbpm)-len(ssr_predict)
        # if minum==0:
        #     ssr_predict = ssr_predict.to_numpy().reshape(-1)
        # else:
        #     ssr_predict = ssr_predict[:minum].to_numpy().reshape(-1)

        # ssr_predict = np.float32(ssr_predict)
        # MSE = mean_squared_error(T_medianbpm, P_medianbpm)
        # RMSE = math.sqrt(MSE)
        # corr = scipy.stats.pearsonr(T_medianbpm, P_medianbpm)
        # print(corr)
        # print(RMSE)
        
        ###############專題生data測試的label
        # bpmlabel = pd.read_csv('/data/Users/huangkaichun/專題生data/{}/{}_bpm.csv'.format(ex,exnum),header=None,index_col=0)
        # minum = len(bpmlabel)-len(P_medianbpm)
        # print(minum)
        # front = int(minum/2)
        # back = minum-front
        # if minum==0:
        #     bpmlabel = bpmlabel.to_numpy().reshape(-1)
        # else:
        #     bpmlabel = bpmlabel[int(front):-int(back)].to_numpy().reshape(-1)

        # bpmlabel = np.float32(bpmlabel)
        # T_medianbpm = bpmlabel
        # P_medianbpm = P_medianbpm[:]
        # MSE = mean_squared_error(T_medianbpm, P_medianbpm)
        # RMSE = math.sqrt(MSE)
        # corr = scipy.stats.pearsonr(T_medianbpm, P_medianbpm)
        # MAE =  mean_absolute_error(T_medianbpm, P_medianbpm)
        # print(corr)
        # print('RMSE：'+str(RMSE))
        # print('MAE：'+str(MAE))
        # ###################################

        # # 寫correlation 和 rmse檔
        # if os.path.exists('img//bpmrmse.csv'):
        #     old = pd.read_csv('img//bpmrmse.csv')
        #     pd.concat([old,pd.DataFrame([RMSE])],axis=1).to_csv('img//bpmrmse.csv',float_format='%.3f')
        #     old = pd.read_csv('img//bpmcorr.csv')
        #     pd.concat([old,pd.DataFrame([corr])],axis=0).to_csv('img//bpmcorr.csv',float_format='%.3f')
        #     old = pd.read_csv('img//bpmMAE.csv')
        #     pd.concat([old,pd.DataFrame([MAE])],axis=1).to_csv('img//bpmMAE.csv',float_format='%.3f')
        # else:
        #     pd.DataFrame([RMSE]).to_csv('img//bpmrmse.csv',float_format='%.3f')
        #     pd.DataFrame([corr]).to_csv('img//bpmcorr.csv',float_format='%.3f')
        #     pd.DataFrame([MAE]).to_csv('img//bpmMAE.csv',float_format='%.3f')
        # # MSE = mean_squared_error(T_medianbpm, ssr_predict)
        # # RMSE = math.sqrt(MSE)
        # # print(scipy.stats.pearsonr(T_medianbpm, ssr_predict))
        # # print(RMSE)
        # if test_data=='UBFC':
        #     plt.subplot(311)
        #     # plt.plot(np.concatenate((pulse_pred[0:180],pulse_pred[1080:1260],pulse_pred[2160:2340],pulse_pred[3240:3420],pulse_pred[4420:4600]),axis=0),label = 'predict')   
        #     plt.plot(pulse_pred[:256],label = 'predict')   
        #     # plt.plot([height]*900,linestyle = 'dotted',label = 'threshold')
        #     plt.legend(loc = 'lower left')
        #     plt.title('Pulse Prediction')
        #     plt.subplot(312)
        #     #label = pd.read_csv(path_label,header=None)
        #     #label = np.array(label).reshape(-1)
        #     # label = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(label))
        #     # plt.plot(label[0:900],label = 'true',color='darkorange')    
        #     plt.plot(label[-256:],label = 'true',color='darkorange')    
        #     plt.legend(loc = 'lower left')
        #     plt.ylim(-2,2)
        #     plt.subplot(313)
        #     plt.plot(P_medianbpm,label = 'predict')
        #     # plt.plot(ssr_predict,label = '2SR',color='green')
        #     plt.plot(T_medianbpm,label = 'true',color='darkorange')
        #     plt.ylim(50,180)
        #     plt.title('BPM')
        #     plt.xlabel('sec')
        #     plt.ylabel('BPM')
        #     plt.legend(loc = 'upper left')
        #     plt.savefig('img/{}.png'.format(num),dpi=300)
        #     plt.clf()
        # else:
        #     plt.subplot(211)
        #     plt.plot(np.concatenate((pulse_pred[0:180],pulse_pred[1080:1260],pulse_pred[2160:2340]),axis=0),label = 'predict') 
        #     plt.xlabel('sec')
        #     plt.legend(loc = 'lower left')
        #     plt.subplot(212)
        #     plt.plot(P_medianbpm,label = 'predict')
        #     plt.plot(T_medianbpm,label = 'true',color='darkorange')
        #     plt.xlabel('sec  \nrmse:{}'.format(RMSE))
        #     plt.ylabel('BPM')
        #     plt.legend(loc = 'lower left')
        #     plt.savefig('img/{}_{}.png'.format(ex,exnum),dpi=300,bbox_inches='tight')
        #     plt.clf()
        break
    return (P_medianbpm)
              


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--sampling_rate', type=int, default = 30, help='sampling rate of your video')
    parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
    args = parser.parse_args()

    predict_vitals(args)

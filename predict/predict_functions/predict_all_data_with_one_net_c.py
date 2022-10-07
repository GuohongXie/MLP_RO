import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from function_predict import function_predict



def predict_all_data_with_one_net_c(source_csv_folder, target_csv_folder, model_c_used, using_gpu=True, which_gpu=0, log_path = ''):
    # 此函数将xyz未经归一化的csv测试数据作为source_csv_folder
    # set device
    if using_gpu:
        cuda_name = "cuda:" + str(which_gpu)
        device = torch.device(cuda_name)
    else:
        device = torch.device("cpu")
    #model_u_used = model_u_used.to(device)
    #model_p_used = model_p_used.to(device)
    model_c_used = model_c_used.to(device)

    #批量生成新模型通道数据
    csv_raw_list = os.listdir(source_csv_folder)
    csv_raw_list.sort(key= lambda x:int(x[:-4]))
    for i in range(len(csv_raw_list)):
        with open(log_path, "a") as log_file:
            print("begin to process %dth csv" %(i), file=log_file)
            log_file.close()
        print("begin to process %dth csv" %(i))
        csv_pred = pd.read_csv(source_csv_folder + csv_raw_list[i],\
                                header = None,\
                                names=['x','y','z','D1A','D2A','D1B','D2B','angle','u','Dtot','U','P','C'],\
                                encoding="utf8")
        
        csv_temp = pd.read_csv(source_csv_folder +  csv_raw_list[i],\
                                header = None,\
                                names=['x','y','z','D1A','D2A','D1B','D2B','angle','u','Dtot','U','P','C'],\
                                encoding="utf8")
        
        #csv_pred.dropna(inplace=True)
        csv_temp[['x']] = preprocessing.MinMaxScaler().fit_transform(csv_temp[['x']])
        csv_temp[['y']] = preprocessing.MinMaxScaler().fit_transform(csv_temp[['y']])
        csv_temp[['z']] = preprocessing.MinMaxScaler().fit_transform(csv_temp[['z']])
        csv_temp['x'] = csv_temp['x'].round(5)
        csv_temp['y'] = csv_temp['y'].round(5)
        csv_temp['z'] = csv_temp['z'].round(5)
        csv_temp["D1A"] = (csv_temp["D1A"]-0.0003)/0.0003
        csv_temp["D2A"] = (csv_temp["D2A"]-0.0002)/0.0001
        csv_temp["D1B"] = (csv_temp["D1B"]-0.0003)/0.0003
        csv_temp["D2B"] = (csv_temp["D2B"]-0.0002)/0.0001
        csv_temp["angle"] = (csv_temp["angle"]-30)/90
        csv_temp["u"] = (csv_temp["u"]-0.045086)/0.148104
        csv_temp["Dtot"] = (csv_temp["Dtot"]-0.0005)/0.0006
    
        for j in range(len(csv_pred)):
            #csv_pred.iloc[j, 10]  = function_predict(csv_temp.values[j,:-3].astype('float32'), model_used_2=model_u_used, device_2=device, using_gpu_2=using_gpu)[0]*0.62367
            #csv_pred.iloc[j, 11]  = function_predict(csv_temp.values[j,:-3].astype('float32'), model_used_2=model_p_used, device_2=device, using_gpu_2=using_gpu)[0]*2224.699 - 52.899
            csv_pred.iloc[j, 12]  = function_predict(csv_temp.values[j,:-3].astype('float32'), model_used_2=model_c_used, device_2=device, using_gpu_2=using_gpu)[0]*1282.6 + 610
            
        csv_pred['U'] = csv_pred['U'].round(8)
        csv_pred['P'] = csv_pred['P'].round(4)
        csv_pred['C'] = csv_pred['C'].round(3)

        csv_pred.to_csv(target_csv_folder + csv_raw_list[i], header=None, index=False, encoding="utf8")
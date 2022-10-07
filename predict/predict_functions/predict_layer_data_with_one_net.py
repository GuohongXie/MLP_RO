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



def predict_layer_data_with_one_net(source_csv_folder, target_csv_folder, model_used, using_gpu=True, which_gpu=0, log_path = ''):
    # set device
    if using_gpu:
        cuda_name = "cuda:" + str(which_gpu)
        device = torch.device(cuda_name)
    else:
        device = torch.device("cpu")
    model_used = model_used.to(device)

    x_min_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_max_minus_min_list = [0.026853, 0.026853, 0.023315, 0.023315, 0.020659, 0.020659, 0.017495, 0.0139, 0.0139, 0.020659]
    y_min_list = [-0.00071952, -0.00071952, -0.0015141, -0.0015141, -0.0018602, -0.0018602, -0.0021605, -0.0024076, -0.0024076, -0.0018602]
    y_max_minus_min_list = [0.00143904, 0.00143904, 0.0030282, 0.0030282, 0.0037204, 0.0037204, 0.004321, 0.0048152, 0.0048152, 0.0037204]
    z_min_list = [-0.0004021, -0.00038706, -0.00048366, -0.00048366, -0.000545, -0.0003575, -0.0004475, -0.00039346, -0.00032, -0.00044651]
    z_max_minus_min_list = [0.00080425, 0.00077415, 0.00096755, 0.00096755, 0.00109, 0.000715, 0.000895, 0.00078668, 0.00064, 0.00089293]


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
        csv_pred['x'] = csv_pred['x']*x_max_minus_min_list[i] + x_min_list[i]
        csv_pred['y'] = csv_pred['y']*y_max_minus_min_list[i] + y_min_list[i]
        csv_pred['z'] = csv_pred['z']*z_max_minus_min_list[i] + z_min_list[i]
        csv_pred['x'] = csv_pred['x'].round(8)
        csv_pred['y'] = csv_pred['y'].round(8)
        csv_pred['z'] = csv_pred['z'].round(8)
        csv_temp = pd.read_csv(source_csv_folder +  csv_raw_list[i],\
                                header = None,\
                                names=['x','y','z','D1A','D2A','D1B','D2B','angle','u','Dtot','U','P','C'],\
                                encoding="utf8")
                           
        #csv_pred.dropna(inplace=True)
        csv_temp["D1A"] = (csv_temp["D1A"]-0.0003)/0.0003
        csv_temp["D2A"] = (csv_temp["D2A"]-0.0002)/0.0001
        csv_temp["D1B"] = (csv_temp["D1B"]-0.0003)/0.0003
        csv_temp["D2B"] = (csv_temp["D2B"]-0.0002)/0.0001
        csv_temp["angle"] = (csv_temp["angle"]-30)/90
        csv_temp["u"] = (csv_temp["u"]-0.045086)/0.148104
        csv_temp["Dtot"] = (csv_temp["Dtot"]-0.0005)/0.0006
    
        for j in range(len(csv_pred)):
            U_P_C_predict_list = function_predict(csv_temp.values[j,:-3].astype('float32'), model_used_2=model_used, device_2=device, using_gpu_2=using_gpu)
            csv_pred.iloc[j, 10] = U_P_C_predict_list[0][0]*0.62367
            csv_pred.iloc[j, 11] = U_P_C_predict_list[0][1]*2224.699 - 52.899
            csv_pred.iloc[j, 12] = U_P_C_predict_list[0][2]*1282.6 + 610

        csv_pred['U'] = csv_pred['U'].round(8)
        csv_pred['P'] = csv_pred['P'].round(4)
        csv_pred['C'] = csv_pred['C'].round(3)
        
        
        csv_pred.to_csv(target_csv_folder + csv_raw_list[i], header=None, index=False, encoding="utf8")
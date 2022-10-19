# predict U P C
#倒入必要的包
import os
import sys
import time
import ntpath
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

os.chdir(sys.path[0])
sys.path.append(os.path.realpath("../.."))
sys.path.append(os.path.realpath(".."))
sys.path.append(os.path.realpath("../predict_functions"))
from models import *
from predict_functions import *


#设置随机数种子
torch.manual_seed(42)    # reproducible

# set working directory
os.chdir(sys.path[0])
print(os.getcwd())

this_file_name = ntpath.basename(os.path.abspath(__file__))
this_file_name = this_file_name[:-3]


#设置网络参数路径
# model_state_dict_path_u_p_c = r"../../results/model_parameters/in/in_u_p_c_4layers_128_64_32_1_Relu_BZ64_LR1e-4_WD1e-7_StepLRSchedule_10_5e-1_noBN_noResnet_noLastRelu_epoch12.pth"
model_state_dict_path_u = r"../../results/model_parameters/all/all_u_s_4layers_128_64_32_1_Relu_BZ32_LR1e-4_WD1e-6_noStepLRSchedule_noBN_noResnet_noLastRelu_epoch9.pth"
model_state_dict_path_p = r"../../results/model_parameters/all/all_p_s_4layers_128_64_32_1_Relu_BZ32_LR1e-4_WD1e-6_noStepLRSchedule_noBN_noResnet_noLastRelu_epoch6.pth"
model_state_dict_path_c = r"../../results/model_parameters/all/all_c_s_4layers_128_64_32_1_Relu_BZ32_LR1e-4_WD1e-6_noStepLRSchedule_noBN_noResnet_noLastRelu_epoch15.pth"
csv_raw_folder = r"../../../data/test_data/"
csv_pred_folder = r"../../../data/predict_data/all/" + this_file_name + "/"
print_log_path = r"../../results/predict_processing_log/all/" + this_file_name + ".txt"

folder_temp = os.path.abspath(csv_pred_folder)
if not os.path.exists(folder_temp):
    os.makedirs(folder_temp)

# 实体化网络并导入对应网络参数
model_u = MLP_128_64_32_1(10)
model_p = MLP_128_64_32_1(10)
model_c = MLP_128_64_32_1(10)

model_u.load_state_dict(torch.load(model_state_dict_path_u))
model_p.load_state_dict(torch.load(model_state_dict_path_p))
model_c.load_state_dict(torch.load(model_state_dict_path_c))



start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
with open(print_log_path, "a") as log_file:
    print("start_time is: %s" %(start_time), file=log_file)
    print("\n", file=log_file)
    log_file.close()
print("start_time is: %s" %(start_time))

#predict_layer_data_with_one_net(csv_raw_folder, csv_pred_folder, model_u_p_c, using_gpu=True, which_gpu=0, log_path = print_log_path)
predict_all_data_single_minmax_with_three_net(csv_raw_folder, csv_pred_folder, model_u, model_p, model_c, log_path = print_log_path)


end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
with open(print_log_path, "a") as log_file:
    print("end_time is: %s" %(end_time), file=log_file)
    print("all done, your predicts are ready", file=log_file)
    print("\n", file=log_file)
    log_file.close()
print("end_time is: %s" %(end_time))
print("all done, your predicts are ready")
#pytorch mlp for regression
import os
import sys
import re
import ntpath
import time
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import preprocessing


def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None

def get_RMSE(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def SMAPE(y_true, y_pred):
    return sum([(abs(x - y)*2)/(abs(x)+abs(y)) for x, y in zip(y_true, y_pred)]) / len(y_true)
    #return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100



# train the model
def evaluate_single_model_file_RMSE_loss(raw_csv_path, predict_csv_path, print_log_file_path=''):
    csv_ordinal_list = re.findall(r"\d+", raw_csv_path)
    csv_ordinal_str = csv_ordinal_list[-1]

    df_raw = pd.read_csv(raw_csv_path,\
                         header = None,\
                         names=['x','y','z','D1A','D2A','D1B','D2B','angle','Qtot','Dtot','U','P','C'],\
                         encoding="utf8")

    U_raw_list = df_raw['U'].values.tolist()
    P_raw_list = df_raw['P'].values.tolist()
    C_raw_list = df_raw['C'].values.tolist()
    U_raw_mean = sum(U_raw_list)/len(U_raw_list)
    P_raw_mean = sum(P_raw_list)/len(P_raw_list)
    C_raw_mean = sum(C_raw_list)/len(C_raw_list)
 

    df_predict = pd.read_csv(predict_csv_path,\
                         header = None,\
                         names=['x','y','z','D1A','D2A','D1B','D2B','angle','Qtot','Dtot','U','P','C'],\
                         encoding="utf8")
    
    U_predict_list = df_predict['U'].values.tolist()
    P_predict_list = df_predict['P'].values.tolist()
    C_predict_list = df_predict['C'].values.tolist()


    with open(print_log_file_path, "a") as log_file:
        print("\n", file=log_file)
        print("start to processing the %s_csv: "%csv_ordinal_str, file=log_file)
        print("the lenth of %s_csv_df_raw is %d"%(csv_ordinal_str, len(df_raw)), file=log_file)
        print("the lenth of %s_csv_df_predict is %d"%(csv_ordinal_str, len(df_predict)), file=log_file)
        log_file.close()

   
    U_R2 = r2_score(U_raw_list, U_predict_list)
    P_R2 = r2_score(P_raw_list, P_predict_list)
    C_R2 = r2_score(C_raw_list, C_predict_list)
    U_MSE = mean_squared_error(U_raw_list, U_predict_list)
    P_MSE = mean_squared_error(P_raw_list, P_predict_list)
    C_MSE = mean_squared_error(C_raw_list, C_predict_list)
    U_MAE = mean_absolute_error(U_raw_list, U_predict_list)
    P_MAE = mean_absolute_error(P_raw_list, P_predict_list)
    C_MAE = mean_absolute_error(C_raw_list, C_predict_list)
    U_SMAPE = SMAPE(U_raw_list, U_predict_list)
    P_SMAPE = SMAPE(P_raw_list, P_predict_list)
    C_SMAPE = SMAPE(C_raw_list, C_predict_list)

    with open(print_log_file_path, "a") as log_file:
        # evaluate the model
        print("U_R2_loss: %.9f,  U_MSE_loss: %.9f,  U_RMSE_loss: %.9f,  U_RRMSE_loss: %.9f,  U_MAE_loss:%.9f,  U_RMAE_loss:%.9f,  U_SMAPE_loss:%.9f" %(U_R2, U_MSE, np.sqrt(U_MSE), (np.sqrt(U_MSE))/U_raw_mean, U_MAE, U_MAE/U_raw_mean, U_SMAPE), file=log_file)
        print("P_R2_loss: %.9f,  P_MSE_loss: %.9f,  P_RMSE_loss: %.9f,  P_RRMSE_loss: %.9f,  P_MAE_loss:%.9f,  P_RMAE_loss:%.9f,  P_SMAPE_loss:%.9f" %(P_R2, P_MSE, np.sqrt(P_MSE), (np.sqrt(P_MSE))/P_raw_mean, P_MAE, P_MAE/P_raw_mean, P_SMAPE), file=log_file)
        print("C_R2_loss: %.9f,  C_MSE_loss: %.9f,  C_RMSE_loss: %.9f,  C_RRMSE_loss: %.9f,  C_MAE_loss:%.9f,  C_RMAE_loss:%.9f,  C_SMAPE_loss:%.9f" %(C_R2, C_MSE, np.sqrt(C_MSE), (np.sqrt(C_MSE))/C_raw_mean, C_MAE, C_MAE/C_raw_mean, C_SMAPE), file=log_file)
        log_file.close()
    
    return None

if __name__=="__main__":
    # set working directory as the path of the code file
    os.chdir(sys.path[0])
    print(os.getcwd())

    test_data_folder  = r'../../data/test_data/' 
    predict_data_folder = r'../../data/predict_data/all/all_threeNet_single_minmax_mlp_128_64_32_1_struct_0x6/'
    print_log_path = r"../results/data_processing_log/evaluate_test_set_single_RMSE_loss_0x6.txt"


    # record time
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(print_log_path, "a") as log_file:
        print('start time is:', start_time, file=log_file)
        log_file.close()
    
    # train the mod
    raw_csv_list = os.listdir(test_data_folder)
    raw_csv_list.sort(key = lambda x:int(x[:-4]))

    predict_csv_list = os.listdir(test_data_folder)
    predict_csv_list.sort(key = lambda x:int(x[:-4]))
    if (not ( len(raw_csv_list)==len(predict_csv_list))):
        print("the raw_csv_folder and predict_csv_folder differs in nums of files")

    for i in range(len(raw_csv_list)):
        if (not(raw_csv_list[i]==predict_csv_list[i])):
            print("the raw_csv_folder and predict_csv_folder differs in contents of files")
            break
        raw_csv_file_path = test_data_folder + raw_csv_list[i]
        predict_csv_file_path = predict_data_folder + predict_csv_list[i]
        evaluate_single_model_file_RMSE_loss(raw_csv_file_path, predict_csv_file_path, print_log_file_path=print_log_path)

    


    # record time
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(print_log_path, "a") as log_file:
        print('end time is:',end_time, file=log_file)
        print("you are all set", file=log_file)
        log_file.close()

    
   
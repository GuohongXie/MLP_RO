#pytorch mlp for regression
import os
import sys
import ntpath
import time
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score







def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None

def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None




# train the model
def evaluate_single_model_file_rmse_loss(raw_csv_path, predict_csv_path, print_log_file_path=''):
     
    with open(print_log_file_path, "a") as log_file:
        print("\n", file=log_file)
        print("start to calculate the u_test_set_rmse_loss: ", file=log_file)
        print("the lenth of test_dataset is %d"%len(test_dl.dataset), file=log_file)
        log_file.close()

    df_raw = pd.read_csv()

    # evaluate the model
    mse_test = evaluate_model(model, test_dl, device_2=device, using_gpu_2=using_gpu)
    with open(print_log_file_path, "a") as log_file:
        # evaluate the model
        print("R2_test: %.9f, valid_loss_mse: %.9f, valid_loss_rmse: %.9f" %(R2_test, mse_test, np.sqrt(mse_test)), file=log_file)
        log_file.close()
    
    
    


# evaluate the model
def evaluate_model(model, test_dl, device_2="cuda:0", using_gpu_2=True):
    predictions,actuals = list(),list()
    for i,(inputs,targets) in enumerate(test_dl):
        # retrieve numpy array
        y_hat = y_hat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual),1))
        # store
        predictions.append(y_hat)
        actuals.append(actual)
    predictions,actuals = np.vstack(predictions),np.vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals,predictions)
    R2 = r2_score(actuals,predictions)
    return R2, mse



if __name__=="__main__":
    # set working directory as the path of the code file
    os.chdir(sys.path[0])
    print(os.getcwd())

    test_data_folder  = r'../../data/test_data/' 
    predict_data_folder = r'../../data/predict_data/all/all_threeNet_single_minmax_mlp_128_64_32_1_struct_0x6/'
    print_log_path = r"../results/data_processing_log/evaluate_test_set_single_rmse_loss.txt"


    # record time
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(print_log_path, "a") as log_file:
        print("\n", file=log_file)
        print("start to evaluate_test_set_rmse_loss", file=log_file)
        print('start time is:', start_time, file=log_file)
        log_file.close()
    
    # train the mod

    


    # record time
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(print_log_path, "a") as log_file:
        print('end time is:',end_time, file=log_file)
        print("you are all set", file=log_file)
        log_file.close()

    
   
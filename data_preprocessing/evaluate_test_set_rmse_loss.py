#pytorch mlp for regression
import os
import sys
import ntpath
import time
import torch
import torch.utils.data
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

os.chdir(sys.path[0])
sys.path.append(os.path.realpath(".."))
from models import *
from csv_datasets import *


torch.manual_seed(42)    # reproducible



# train the model
def evaluate_test_set_rmse_loss(model, test_dataset, model_state_dict_path,\
                                batch__size = 32, using_gpu=True, which_gpu=0, \
                                var_name="evaluate_test_set_rmse_loss" , print_log_file_path=''):
    # set device
    if using_gpu:
        cuda_name = "cuda:" + str(which_gpu)
        device = torch.device(cuda_name)
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # load model_parameters
    model.load_state_dict(torch.load(model_state_dict_path))
    
    # dataloader
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch__size, shuffle=False)
    with open(print_log_file_path, "a") as log_file:
        print("\n", file=log_file)
        print("start to calculate the %s: "%var_name, file=log_file)
        print("the lenth of test_dataset is %d"%len(test_dl.dataset), file=log_file)
        log_file.close()

    # evaluate the model
    R2_test, mse_test = evaluate_model(model, test_dl, device_2=device, using_gpu_2=using_gpu)
    with open(print_log_file_path, "a") as log_file:
        # evaluate the model
        print("R2_test: %.9f, valid_loss_mse: %.9f, valid_loss_rmse: %.9f" %(R2_test, mse_test, np.sqrt(mse_test)), file=log_file)
        log_file.close()
    
    
    


# evaluate the model
def evaluate_model(model, test_dl, device_2="cuda:0", using_gpu_2=True):
    predictions,actuals = list(),list()
    for i,(inputs,targets) in enumerate(test_dl):
        inputs = inputs.to(device_2)
        # evaluate the model on the test set
        y_hat = model(inputs)
        if using_gpu_2:
            y_hat = y_hat.cpu()
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


    #设置参数路径
    model_state_dict_path_u = r"../results/model_parameters/all/all_u_s_4layers_128_64_32_1_Relu_BZ32_LR1e-4_WD1e-6_noStepLRSchedule_noBN_noResnet_noLastRelu_epoch1.pth"
    model_state_dict_path_p = r"../results/model_parameters/all/all_p_s_4layers_128_64_32_1_Relu_BZ32_LR1e-4_WD1e-6_noStepLRSchedule_noBN_noResnet_noLastRelu_epoch6.pth"
    model_state_dict_path_c = r"../results/model_parameters/all/all_c_s_4layers_128_64_32_1_Relu_BZ32_LR1e-4_WD1e-6_noStepLRSchedule_noBN_noResnet_noLastRelu_epoch4.pth"
    test_data_path  = r'../../data/train_and_valid_merged_csv_single_minmax_xyz/test_data_csv/test_data_csv_10.csv' 
    print_log_path = r"../results/data_processing_log/evaluate_test_set_rmse_loss.txt"

    # 实体化网络
    model_u  = MLP_128_64_32_1(10)
    model_p  = MLP_128_64_32_1(10)
    model_c  = MLP_128_64_32_1(10)
    

    # prepare the data 
    
    test_dataset_u  = U_Valid_Dataset_single_minMaxScaler(test_data_path)
    test_dataset_p  = P_Valid_Dataset_single_minMaxScaler(test_data_path)
    test_dataset_c  = C_Valid_Dataset_single_minMaxScaler(test_data_path)

    # record time
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(print_log_path, "a") as log_file:
        print("\n", file=log_file)
        print("start to evaluate_test_set_rmse_loss", file=log_file)
        print('start time is:', start_time, file=log_file)
        log_file.close()
    
    # train the model
    evaluate_test_set_rmse_loss(model_u, test_dataset_u, model_state_dict_path_u, \
                                batch__size = 32, using_gpu=True, which_gpu=0, \
                                var_name="u_test_set_rmse_loss" , print_log_file_path=print_log_path)
    
    evaluate_test_set_rmse_loss(model_p, test_dataset_p, model_state_dict_path_p, \
                                batch__size = 32, using_gpu=True, which_gpu=0, \
                                var_name="p_test_set_rmse_loss" , print_log_file_path=print_log_path)

    evaluate_test_set_rmse_loss(model_c, test_dataset_c, model_state_dict_path_c, \
                                batch__size = 32, using_gpu=True, which_gpu=0, \
                                var_name="c_test_set_rmse_loss" , print_log_file_path=print_log_path)
    

    # record time
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(print_log_path, "a") as log_file:
        print('end time is:',end_time, file=log_file)
        print("you are all set", file=log_file)
        log_file.close()

    
   
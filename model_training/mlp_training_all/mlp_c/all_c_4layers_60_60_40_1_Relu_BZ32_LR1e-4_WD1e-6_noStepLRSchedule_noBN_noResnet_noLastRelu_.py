#pytorch mlp for regression
import os
import sys
import ntpath
import time
import torch

from torch.autograd import Variable
from torchvision import transforms

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


os.chdir(sys.path[0])
sys.path.append(os.path.realpath("../../.."))
from models import *
from csv_datasets import *


torch.manual_seed(707)    # reproducible



# train the model
def train_model(train_dataset_1, valid_dataset_1, model_1, \
                epoch_num=40, batch__size = 8192, learning_rate=1e-4, weight__decay=1e-7, \
                using_gpu=True, which_gpu=0, \
                resume_num=0, load_model=False, model_parameters_folder='../../../results/model_parameters/',
                print_folder='../../../results/training_process_output/', \
                load_other_model=False, other_model_parameter_path = "" ):
    '''
    resume_num 指的是这次开始训练的epoch, 即上一次训练最后一个保存的epoch+1, 这里的epoch从0开始
    
    '''
    # dataloader
    train_dl = torch.utils.data.DataLoader(train_dataset_1, batch_size=batch__size, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_dataset_1, batch_size=batch__size, shuffle=False)
    

    # set device
    if using_gpu:
        cuda_name = "cuda:" + str(which_gpu)
        device = torch.device(cuda_name)
    else:
        device = torch.device("cpu")
    model_1 = model_1.to(device)

    # load model_parameters
    if(load_model and load_other_model):
        print("load_model and load_other_model are in conflict")
        sys.exit(0)
    file_name = ntpath.basename(os.path.abspath(__file__))
    file_name = file_name[:-4]
    print_log_file = print_folder + file_name + ".txt"
    with open(print_log_file, "a") as log_file:
        print('start time is:',start_time, file=log_file)
        print("the lenth of train_dataset is %d"%len(train_dl.dataset), file=log_file)
        print("the lenth of valid_dataset is %d"%len(valid_dl.dataset), file=log_file)
        log_file.close()
    if load_model:
        load_model_path = model_parameters_folder + file_name + "_epoch" + str(resume_num-1) + ".pth"
        model_1.load_state_dict(torch.load(load_model_path))
    if load_other_model:
        model_1.load_state_dict(torch.load(other_model_parameter_path))
    
    #define the optimization
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate, weight_decay=weight__decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)

    # enumerate epochs
    for epoch in range(resume_num, resume_num+epoch_num):
        # enumerate mini batches
        for i,(inputs,targets) in enumerate(train_dl):
            inputs,targets = inputs.to(device), targets.to(device)
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            y_hat = model_1(inputs) 
            # calculate loss
            loss = criterion(y_hat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        #scheduler.step()

        # evaluate the model
        R2_valid, mse_valid = evaluate_model(valid_dl, model_1, device_2=device, using_gpu_2=using_gpu)
        with open(print_log_file, "a") as log_file:
            # show the loss
            print("epoch: %d, loss = %.9f" %(epoch,loss), file=log_file)
            # evaluate the model
            print("R2_valid: %.9f, valid_loss_mse: %.9f, valid_loss_rmse: %.9f" %(R2_valid, mse_valid, np.sqrt(mse_valid)), file=log_file)
            log_file.close()
        #save model parameters
        save_model_path = model_parameters_folder + file_name + "_epoch" + str(epoch) + ".pth"
        torch.save(model_1.state_dict(), save_model_path)
    
    end_time_1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(print_log_file, "a") as log_file:
        print('end time is:',end_time_1, file=log_file)
        print("you are all set", file=log_file)
        log_file.close()


# evaluate the model
def evaluate_model(valid_dl_2,model_2, device_2, using_gpu_2=True):
    predictions,actuals = list(),list()
    for i,(inputs,targets) in enumerate(valid_dl_2):
        inputs = inputs.to(device_2)
        # evaluate the model on the test set
        y_hat = model_2(inputs)
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

    # record time
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('start time is:',start_time)

    # set working directory as the path of the code file
    os.chdir(sys.path[0])
    print(os.getcwd())

    # prepare the data 
    train_data_path = r'../../../../data/merged_csv/train_data_csv_650/train_data_csv_650.csv' 
    valid_data_path = r'../../../../data/merged_csv/valid_data_csv_74/valid_data_csv_74.csv'
    

    train_dataset = C_Train_Dataset_minMaxScaler_all(train_data_path)
    valid_dataset = C_Valid_Dataset_minMaxScaler_all(valid_data_path)


    
    # define the network
    model = MLP_128_64_32_1(10)
    
    # train the model
    train_model(train_dataset, valid_dataset, model, \
                epoch_num=20, batch__size = 32, learning_rate=1e-4, weight__decay=1e-7, \
                using_gpu=True, which_gpu=0, \
                resume_num=0, load_model=False, model_parameters_folder='../../../results/model_parameters/all/',
                print_folder='../../../results/training_process_output/all/', \
                load_other_model=False, other_model_parameter_path = "../../../results/model_parameters/in/in_c_4layers_128_64_32_1_Relu_BZ64_LR1e-4_WD1e-7_StepLRSchedule_10_5e-1_noBN_noResnet_noLastRelu_epoch2.pth" )

    # record time
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('end time is:', end_time)
    print("you are all set")

    
   
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

import torch.utils.data
import pandas as pd
import sklearn



torch.manual_seed(42)    # reproducible






#dataset definition
class P_Train_Dataset_minMaxScaler_all(torch.utils.data.Dataset):
    #load the dataset
    def __init__(self,path_1):
        # load the csv file as a dataframe
        train_csv = pd.read_csv(path_1,\
                         header = None,\
                         names=['x','y','z','D1A','D2A','D1B','D2B','angle','Qtot','Dtot','U','P','C'],\
                         encoding="utf8")  # df的type是<class 'pandas.core.frame.DataFrame'>
        #df.dropna(inplace=True) #这一行不需要
        train_csv["x"]     = (train_csv["x"])/0.026853
        train_csv["y"]     = (train_csv["y"]+0.0024076)/0.0048156
        train_csv["z"]     = (train_csv["z"]+0.000545)/0.00109
        train_csv["D1A"]   = (train_csv["D1A"]-0.0003)/0.0003
        train_csv["D2A"]   = (train_csv["D2A"]-0.0002)/0.0001
        train_csv["D1B"]   = (train_csv["D1B"]-0.0003)/0.0003
        train_csv["D2B"]   = (train_csv["D2B"]-0.0002)/0.0001
        train_csv["angle"] = (train_csv["angle"]-30)/90
        train_csv["Qtot"]  = (train_csv["Qtot"]-45.373)/327.707
        train_csv["Dtot"]  = (train_csv["Dtot"]-0.0005)/0.0006
        train_csv["U"]     = (train_csv["U"])/0.6308
        train_csv["P"]     = (train_csv["P"]+52.899)/2224.699
        train_csv["C"]     = (train_csv["C"]-600)/1292.6
        #df = sklearn.preprocessing.MinMaxScaler().fit_transform(df)  #这一行把df的type变成了<class 'numpy.ndarray'>，现在不需要这一行
        # store the inputs and outputs
        self.X = train_csv.values[:, :-3].astype('float32')   #这一行之后同样，X的type为<class 'numpy.ndarray'>
        self.y = train_csv.values[:, -2].astype('float32')
        # ensure target has the right shape
        self.y = self.y.reshape(len(self.y),1)
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
    # get a row at an index
    def __getitem__(self,idx):
        return [self.X[idx],self.y[idx]]



class P_Valid_Dataset_minMaxScaler_all(torch.utils.data.Dataset):
    #load the dataset
    def __init__(self,path_2):
        # load the csv file as a dataframe
        valid_csv = pd.read_csv(path_2,\
                                header = None,\
                                names=['x','y','z','D1A','D2A','D1B','D2B','angle','Qtot','Dtot','U','P','C'],\
                                encoding="utf8")
        #valid_csv.dropna(inplace=True)
        valid_csv["x"]     = (valid_csv["x"])/0.026853
        valid_csv["y"]     = (valid_csv["y"]+0.0024076)/0.0048156
        valid_csv["z"]     = (valid_csv["z"]+0.000545)/0.00109
        valid_csv["D1A"]   = (valid_csv["D1A"]-0.0003)/0.0003
        valid_csv["D2A"]   = (valid_csv["D2A"]-0.0002)/0.0001
        valid_csv["D1B"]   = (valid_csv["D1B"]-0.0003)/0.0003
        valid_csv["D2B"]   = (valid_csv["D2B"]-0.0002)/0.0001
        valid_csv["angle"] = (valid_csv["angle"]-30)/90
        valid_csv["Qtot"]  = (valid_csv["Qtot"]-45.373)/327.707
        valid_csv["Dtot"]  = (valid_csv["Dtot"]-0.0005)/0.0006
        valid_csv["U"]     = (valid_csv["U"])/0.6308
        valid_csv["P"]     = (valid_csv["P"]+52.899)/2224.699
        valid_csv["C"]     = (valid_csv["C"]-600)/1292.6
        # store the inputs and outputs
        self.valX = valid_csv.values[:, :-3].astype('float32')
        self.valy = valid_csv.values[:, -2].astype('float32')
        # ensure target has the right shape
        self.valy = self.valy.reshape(len(self.valy),1)
    # number of rows in the dataset
    def __len__(self):
        return len(self.valX)
    # get a row at an index
    def __getitem__(self,idx):
        return [self.valX[idx],self.valy[idx]]



# model definition
class MLP_60_60_40_1(torch.nn.Module):
    # define model elements
    def __init__(self,n_inputs):
        super(MLP_60_60_40_1,self).__init__()
        # input to first hidden layer
        self.hidden1 = torch.nn.Linear(n_inputs,60)
        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        self.act1 = torch.nn.ReLU()
        # second hidden layer
        self.hidden2 = torch.nn.Linear(60,60)
        torch.nn.init.xavier_uniform_(self.hidden2.weight)
        self.act2 = torch.nn.ReLU()
        #third hidden layer and output
        self.hidden3 = torch.nn.Linear(60,40)
        torch.nn.init.xavier_uniform_(self.hidden3.weight)
        self.act3 = torch.nn.ReLU()
        #forth hidden layer and output
        self.hidden4 = torch.nn.Linear(40,1)
        torch.nn.init.xavier_uniform_(self.hidden4.weight)
        

    # forward propagate input
    def forward(self,X):
        # input to the first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer 
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        # forth hidden layer and output
        X = self.hidden4(X)
        return X



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
    train_data_path = r'../../../../data/train_and_valid_merged_csv_Qtot/train_data_csv/train_data_csv_650.csv' 
    valid_data_path = r'../../../../data/train_and_valid_merged_csv_Qtot/valid_data_csv/valid_data_csv_74.csv'
    

    train_dataset = P_Train_Dataset_minMaxScaler_all(train_data_path)
    valid_dataset = P_Valid_Dataset_minMaxScaler_all(valid_data_path)


    
    # define the network
    model = MLP_60_60_40_1(10)
    
    # train the model
    train_model(train_dataset, valid_dataset, model, \
                epoch_num=20, batch__size = 32, learning_rate=1e-5, weight__decay=1e-6, \
                using_gpu=True, which_gpu=0, \
                resume_num=0, load_model=False, model_parameters_folder='../../../results/model_parameters/all/',
                print_folder='../../../results/training_process_output/all/', \
                load_other_model=False, other_model_parameter_path = "../../../results/model_parameters/in/in_c_4layers_128_64_32_1_Relu_BZ64_LR1e-4_WD1e-7_StepLRSchedule_10_5e-1_noBN_noResnet_noLastRelu_epoch2.pth" )

    # record time
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('end time is:', end_time)
    print("you are all set")

    
   
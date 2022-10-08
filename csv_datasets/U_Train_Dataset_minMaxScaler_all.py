import torch

import numpy as np
import pandas as pd
from sklearn import preprocessing


#dataset definition
class U_Train_Dataset_minMaxScaler_all(torch.utils.data.Dataset):
    #load the dataset
    def __init__(self,path_1):
        # load the csv file as a dataframe
        train_csv = pd.read_csv(path_1,\
                         header = None,\
                         names=['x','y','z','D1A','D2A','D1B','D2B','angle','u','Dtot','U','P','C'],\
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
        train_csv["u"]     = (train_csv["u"]-0.045905)/0.147285
        train_csv["Dtot"]  = (train_csv["Dtot"]-0.0005)/0.0006
        train_csv["U"]     = (train_csv["U"])/0.6308
        train_csv["P"]     = (train_csv["P"]+52.899)/2224.699
        train_csv["C"]     = (train_csv["C"]-600)/1292.6
        #df = sklearn.preprocessing.MinMaxScaler().fit_transform(df)  #这一行把df的type变成了<class 'numpy.ndarray'>，现在不需要这一行
        # store the inputs and outputs
        self.X = train_csv.values[:, :-3].astype('float32')   #这一行之后同样，X的type为<class 'numpy.ndarray'>
        self.y = train_csv.values[:, -3].astype('float32')
        # ensure target has the right shape
        self.y = self.y.reshape(len(self.y),1)
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
    # get a row at an index
    def __getitem__(self,idx):
        return [self.X[idx],self.y[idx]]
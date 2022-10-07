import torch

import numpy as np
import pandas as pd
from sklearn import preprocessing


#dataset definition
class C_Train_Dataset_minMaxScaler_all(torch.utils.data.Dataset):
    #load the dataset
    def __init__(self,path_1):
        # load the csv file as a dataframe
        train_csv = pd.read_csv(path_1,\
                         header = None,\
                         names=['x','y','z','D1A','D2A','D1B','D2B','angle','u','Dtot','U','P','C'],\
                         encoding="utf8")  # df的type是<class 'pandas.core.frame.DataFrame'>
        #df.dropna(inplace=True) #这一行不需要
        train_csv[["D1A"]] = preprocessing.MinMaxScaler().fit_transform(train_csv[["D1A"]])
        train_csv[["D2A"]] = preprocessing.MinMaxScaler().fit_transform(train_csv[["D2A"]])
        train_csv[["D1B"]] = preprocessing.MinMaxScaler().fit_transform(train_csv[["D1B"]])
        train_csv[["D2B"]] = preprocessing.MinMaxScaler().fit_transform(train_csv[["D2B"]])
        train_csv[["angle"]] = preprocessing.MinMaxScaler().fit_transform(train_csv[["angle"]])
        train_csv[["u"]] = preprocessing.MinMaxScaler().fit_transform(train_csv[["u"]])
        train_csv[["Dtot"]] = preprocessing.MinMaxScaler().fit_transform(train_csv[["Dtot"]])
        train_csv["U"] = (train_csv["U"])/0.62367
        train_csv["P"] = (train_csv["P"] + 52.899)/2224.699
        train_csv["C"] = (train_csv["C"] - 610)/1282.6
        #df = sklearn.preprocessing.MinMaxScaler().fit_transform(df)  #这一行把df的type变成了<class 'numpy.ndarray'>，现在不需要这一行
        # store the inputs and outputs
        self.X = train_csv.values[:, :-3].astype('float32')   #这一行之后同样，X的type为<class 'numpy.ndarray'>
        self.y = train_csv.values[:, -1].astype('float32')
        # ensure target has the right shape
        self.y = self.y.reshape(len(self.y),1)
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
    # get a row at an index
    def __getitem__(self,idx):
        return [self.X[idx],self.y[idx]]
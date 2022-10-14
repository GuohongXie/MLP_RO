import torch

import numpy as np
import pandas as pd
from sklearn import preprocessing


#dataset definition
class P_Valid_Dataset_single_minMaxScaler(torch.utils.data.Dataset):
    #load the dataset
    def __init__(self,path_2):
        # load the csv file as a dataframe
        valid_csv = pd.read_csv(path_2,\
                                header = None,\
                                names=['x','y','z','D1A','D2A','D1B','D2B','angle','Qtot','Dtot','U','P','C'],\
                                encoding="utf8")
        #valid_csv.dropna(inplace=True)
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
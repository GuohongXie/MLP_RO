#pytorch mlp for regression
#import os
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

torch.manual_seed(42)    # reproducible
#device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

#dataset definition
class Train_Dataset(torch.utils.data.Dataset):
    #load the dataset
    def __init__(self,path_1):
        # load the csv file as a dataframe
        df = pd.read_csv(path_1, header=None)
        df.dropna(inplace=True)
        df = sklearn.preprocessing.MinMaxScaler().fit_transform(df)
        # store the inputs and outputs
        self.X = df[:, :-3].astype('float32')
        self.y = df[:, -1].astype('float32')
        # ensure target has the right shape
        self.y = self.y.reshape(len(self.y),1)
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
    # get a row at an index
    def __getitem__(self,idx):
        return [self.X[idx],self.y[idx]]



class Valid_Dataset(torch.utils.data.Dataset):
    #load the dataset
    def __init__(self,path_2):
        # load the csv file as a dataframe
        valid_csv = pd.read_csv(path_2)
        valid_csv.dropna(inplace=True)
        valid_csv["C"] = (valid_csv["C"]-600)/1292.6
        valid_csv["x"] = (valid_csv["x"])/0.026853
        valid_csv["y"] = (valid_csv["y"]+0.0024076)/0.0048156
        valid_csv["z"] = (valid_csv["z"]+0.000545)/0.00109
        valid_csv["D1A"] = (valid_csv["D1A"]-0.0003)/0.0003
        valid_csv["D2A"] = (valid_csv["D2A"]-0.0002)/0.0001
        valid_csv["D1B"] = (valid_csv["D1B"]-0.0003)/0.0003
        valid_csv["D2B"] = (valid_csv["D2B"]-0.0002)/0.0001
        valid_csv["angle"] = (valid_csv["angle"]-30)/90
        valid_csv["Qtot"] = (valid_csv["Qtot"]-45.373)/327.707
        valid_csv["Dtot"] = (valid_csv["Dtot"]-0.0005)/0.0006
        valid_csv["U"] = (valid_csv["U"])/0.6308
        valid_csv["P"] = (valid_csv["P"]+52.899)/2224.699
        # store the inputs and outputs
        self.valX = valid_csv.values[:, :-3].astype('float32')
        self.valy = valid_csv.values[:, -1].astype('float32')
        # ensure target has the right shape
        self.valy = self.valy.reshape(len(self.valy),1)
    # number of rows in the dataset
    def __len__(self):
        return len(self.valX)
    # get a row at an index
    def __getitem__(self,idx):
        return [self.valX[idx],self.valy[idx]]



# model definition
class MLP(torch.nn.Module):
    # define model elements
    def __init__(self,n_inputs):
        super(MLP,self).__init__()
        # input to first hidden layer
        self.hidden1 = torch.nn.Linear(n_inputs,40)
        torch.nn.init.xavier_uniform_(self.hidden1.weight)
        self.act1 = torch.nn.ReLU()
        # second hidden layer
        self.hidden2 = torch.nn.Linear(40,40)
        torch.nn.init.xavier_uniform_(self.hidden2.weight)
        self.act2 = torch.nn.ReLU()
        #third hidden layer and output
        self.hidden3 = torch.nn.Linear(40,25)
        torch.nn.init.xavier_uniform_(self.hidden3.weight)
        self.act3 = torch.nn.ReLU()
        #forth hidden layer and output
        self.hidden4 = torch.nn.Linear(25,1)
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
def train_model(train_dl_form,valid_dl_form,model_form_1):
    #define the optimization
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model_form_1.parameters(), lr=1e-4,weight_decay=1e-6)
    # enumerate epochs
    for epoch in range(50):
        # enumerate mini batches
        for i,(inputs,targets) in enumerate(train_dl_form):
           # inputs,targets = inputs.to(device), targets.to(device)
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            y_hat = model_form_1(inputs) 
            # calculate loss
            loss = criterion(y_hat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        # show the loss
        if epoch % 1 == 0:
            print("epoch: %d, loss = %.9f" %(epoch,loss))
            # evaluate the model
            R2_valid, mse_valid = evaluate_model(valid_dl_form,model_form_1)
            print("R2_valid: %.9f, valid_loss_mse: %.9f, valid_loss_rmse: %.9f" %(R2_valid, mse_valid, np.sqrt(mse_valid)))
        if epoch == 2:
            torch.save(model.state_dict(), r'/home/liu/桌面/GuohonngXie/new_RO_MLN/result/parameters/c_new_test012_epoch3.pth')
        if epoch == 4:
            torch.save(model.state_dict(), r'/home/liu/桌面/GuohonngXie/new_RO_MLN/result/parameters/c_new_test012_epoch5.pth')
        if epoch == 9:
            torch.save(model.state_dict(), r'/home/liu/桌面/GuohonngXie/new_RO_MLN/result/parameters/c_new_test012_epoch10.pth')
        if epoch == 19:
            torch.save(model.state_dict(), r'/home/liu/桌面/GuohonngXie/new_RO_MLN/result/parameters/c_new_test012_epoch20.pth')
        if epoch == 29:
            torch.save(model.state_dict(), r'/home/liu/桌面/GuohonngXie/new_RO_MLN/result/parameters/c_new_test012_epoch30.pth')
        if epoch == 39:
            torch.save(model.state_dict(), r'/home/liu/桌面/GuohonngXie/new_RO_MLN/result/parameters/c_new_test012_epoch40.pth')

# evaluate the model
def evaluate_model(test_dl_form,model_form_2):
    predictions,actuals = list(),list()
    for i,(inputs,targets) in enumerate(test_dl_form):
        # inputs = inputs.to(device)
        # evaluate the model on the test set
        y_hat = model_form_2(inputs)
        # y_hat = y_hat.cpu()
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

    # prepare the data 
    train_data_path = r"/home/liu/桌面/GuohonngXie/new_RO_MLN/data/tarin_data_csv/train_csv.csv"
    valid_data_path = r"/home/liu/桌面/GuohonngXie/new_RO_MLN/data/valid_data_csv/val_csv.csv"
    train_dataset = Train_Dataset(train_data_path)
    valid_dataset = Valid_Dataset(valid_data_path)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

    print(len(train_dl.dataset),len(valid_dl.dataset))
    
    # define the network
    model = MLP(10)#.to(device)
    #model.to(device)
    # train the model
    train_model(train_dl,valid_dl,model)

    #save model parameters
    torch.save(model.state_dict(), r'/home/liu/桌面/GuohonngXie/new_RO_MLN/result/parameters/c_new_test012_epoch50.pth')
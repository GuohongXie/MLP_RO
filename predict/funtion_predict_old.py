# predict U P C
#倒入必要的包
import os
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#设置随机数种子
torch.manual_seed(42)    # reproducible
#device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print("温馨提示：请记得在下一步设置数据和参数路径")


#设置网络参数路径
model_state_dict_path_c = r"/home/liu/桌面/GuohonngXie/new_RO_MLN/result/parameters/c_new_test014_epoch10.pth"
model_state_dict_path_p = r"/home/liu/桌面/GuohonngXie/new_RO_MLN/result/parameters/p_new_test015_epoch10.pth"
model_state_dict_path_u = r"/home/liu/桌面/GuohonngXie/new_RO_MLN/result/parameters/u_new_test014_epoch10.pth"
model_raw_folder = r"/home/liu/桌面/GuohonngXie/new_RO_MLN/predict/model_raw"
model_pred_folder = r"/home/liu/桌面/GuohonngXie/new_RO_MLN/predict/model_pred_U_P_C"


#导入网络结构
class MLP(torch.nn.Module):
    # define model elements
    def __init__(self,n_inputs):
        super(MLP,self).__init__()
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


# 实体化网络并导入对应网络参数
model_c = MLP(10)
model_p = MLP(10)
model_u = MLP(10)
model_c.load_state_dict(torch.load(model_state_dict_path_c))
model_p.load_state_dict(torch.load(model_state_dict_path_p))
model_u.load_state_dict(torch.load(model_state_dict_path_u))


#定义预测函数
def function_predict(row,model):
    # convert row to data
    row = torch.Tensor([row])
    # make prediction
    y_hat = model(row)
    # retrieve numpy array
    y_hat = y_hat.detach().numpy()
    return y_hat


#批量生成新模型通道数据
os.chdir(model_raw_folder)
model_raw_list = os.listdir()
for i in range(len(model_raw_list)):
    model_pred = pd.read_csv(model_raw_folder + '/'+ model_raw_list[i])
    model_pred.dropna(inplace=True)
    model_pred["C"] = (model_pred["C"]-600)/1292.6
    model_pred["x"] = (model_pred["x"])/0.026853
    model_pred["y"] = (model_pred["y"]+0.0024076)/0.0048156
    model_pred["z"] = (model_pred["z"]+0.000545)/0.00109
    model_pred["D1A"] = (model_pred["D1A"]-0.0003)/0.0003
    model_pred["D2A"] = (model_pred["D2A"]-0.0002)/0.0001
    model_pred["D1B"] = (model_pred["D1B"]-0.0003)/0.0003
    model_pred["D2B"] = (model_pred["D2B"]-0.0002)/0.0001
    model_pred["angle"] = (model_pred["angle"]-30)/90
    model_pred["Qtot"] = (model_pred["Qtot"]-45.373)/327.707
    model_pred["Dtot"] = (model_pred["Dtot"]-0.0005)/0.0006
    model_pred["U"] = (model_pred["U"])/0.6308
    model_pred["P"] = (model_pred["P"]+52.899)/2224.699
    for j in range(len(model_pred)):
        model_pred["C"][j] = function_predict(model_pred.values[j,:-3].astype('float32'), model_c)*1292.6+600
        model_pred["C"][j] = round(model_pred["C"][j], 3)
        model_pred["P"][j] = function_predict(model_pred.values[j,:-3].astype('float32'), model_p)*2224.699-52.899
        model_pred["P"][j] = round(model_pred["P"][j], 6)
        model_pred["U"][j] = function_predict(model_pred.values[j,:-3].astype('float32'), model_u)*0.6308
        model_pred["U"][j] = round(model_pred["U"][j], 8)
    model_pred["x"] = (model_pred["x"])*0.026853
    model_pred["y"] = (model_pred["y"])*0.0048156-0.0024076
    model_pred["z"] = (model_pred["z"])*0.00109-0.000545
    model_pred["D1A"] = (model_pred["D1A"])*0.0003+0.0003
    model_pred["D2A"] = (model_pred["D2A"])*0.0001+0.0002
    model_pred["D1B"] = (model_pred["D1B"])*0.0003+0.0003
    model_pred["D2B"] = (model_pred["D2B"])*0.0001+0.0002
    model_pred["angle"] = (model_pred["angle"])*90+30
    model_pred["Qtot"] = (model_pred["Qtot"])*327.707+45.373
    model_pred["Dtot"] = (model_pred["Dtot"])*0.0006+0.0005
    model_pred.to_csv(model_pred_folder + '/pred_'+model_raw_list[i])

print("all done, your predicts are ready")
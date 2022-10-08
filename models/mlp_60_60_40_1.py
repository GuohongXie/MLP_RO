# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# model definition
class MLP_60_60_40_1(nn.Module):
    # define model elements
    def __init__(self,n_inputs):
        super(MLP_60_60_40_1,self).__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(n_inputs,60)
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(60,60)
        nn.init.xavier_uniform_(self.hidden2.weight)
        self.act2 = nn.ReLU()
        #third hidden layer and output
        self.hidden3 = nn.Linear(60,40)
        nn.init.xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.ReLU()
        #forth hidden layer and output
        self.hidden4 = nn.Linear(40,1)
        nn.init.xavier_uniform_(self.hidden4.weight)
        

    # forward propagate input
    def forward(self,X):
        # input to the first hidden layer
        out = self.hidden1(X)
        out = self.act1(out)
        # second hidden layer 
        out = self.hidden2(out)
        out = self.act2(out)
        # third hidden layer and output
        out = self.hidden3(out)
        out = self.act3(out)
        # forth hidden layer and output
        out = self.hidden4(out)
        return out


def test():
    net = MLP_60_60_40_1(10)
    x = torch.randn(10)
    y = net(x)
    print(y.size())


if __name__=="__main__":
    test()
    
   
# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# model definition
class MLP_Resnet_64_64_64_64_64_1(nn.Module):
    # define model elements
    def __init__(self,n_inputs):
        super(MLP_Resnet_64_64_64_64_64_1,self).__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(n_inputs, 64)
        nn.init.xavier_uniform_(self.hidden1.weight)
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(64, 64)
        nn.init.xavier_uniform_(self.hidden2.weight)
        self.act2 = nn.ReLU()
        #third hidden layer and output
        self.hidden3 = nn.Linear(64, 64)
        nn.init.xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.ReLU()
        #forth hidden layer and output
        self.hidden4 = nn.Linear(64, 64)
        nn.init.xavier_uniform_(self.hidden4.weight)
        self.act4 = nn.ReLU()
        self.hidden5 = nn.Linear(64, 64)
        nn.init.xavier_uniform_(self.hidden5.weight)
        self.act5 = nn.ReLU()
        self.hidden6 = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.hidden6.weight)
        #self.act6 = nn.ReLU()
        

    # forward propagate input
    def forward(self,X):
        # input to the first hidden layer
        out = self.hidden1(X)
        out = self.act1(out)
        shortcut_1 = out
        # second hidden layer 
        out = self.hidden2(out)
        out = self.act2(out)
        # third hidden layer and output
        out = self.hidden3(out)
        out += shortcut_1
        out = self.act3(out)
        shortcut_2 = out
        # forth hidden layer and output
        out = self.hidden4(out)
        out = self.act4(out)
        # fifth hidden layer and output
        out = self.hidden5(out)
        out += shortcut_2
        out = self.act5(out)
        # sixth hidden layer and output
        out = self.hidden6(out)
        #out = self.act6(out)
        return out


def test():
    net = MLP_Resnet_64_64_64_64_64_1(10)
    x = torch.randn(10)
    y = net(x)
    print(y.size())


if __name__=="__main__":
    test()
    
   
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable

NetIO = Union[FloatTensor, Variable]


class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.dropout1=  nn.Dropout(p=0.2)
    def forward(self, x: NetIO) -> NetIO:
        # compute the representation for each data point
        x = self.phi.forward(x)

        # sum up the representations
        # here I have assumed that x is 2D and the each row is representation of an input, so the following operation
        # will reduce the number of rows to 1, but it will keep the tensor as a 2D tensor.
        #print("x size before sum",x.size())
        x = torch.sum(x, dim=1, keepdim=True) #sum
        x=x.reshape((x.size()[0],x.size()[2]))
        #print("x size after sum",x.size())
        #x=self.dropout(x)

        # compute the output
        out = self.rho.forward(x)

        return out

#class smallConvPhi():
#    conv1d_res = self.conv1d_params1(to_conv1d)
#    conv1d_res = F.relu(conv1d_res)
#    conv1d_res = self.maxpool1d(conv1d_res)
#    conv1d_res = self.conv1d_params2(conv1d_res)
#    conv1d_res = F.relu(conv1d_res)

#    to_dense = conv1d_res.view(len(graph_sizes), -1)

class SmallMNISTCNNPhi(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc1_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: NetIO) -> NetIO:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2_drop(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        return x

class SmallPhi(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1, hidden_size: int = 1, number_of_layers=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        step=(self.output_size - self.input_size)//2
        print("step phi", step)
        print("input_size phi", self.input_size)

        print("output_size phi", self.output_size)
        self.fc1 = nn.Linear(self.input_size, self.input_size+step)
        self.fc2 = nn.Linear(self.input_size+step, self.output_size)
        #self.fc1 = nn.Linear(self.input_size, self.output_size)
        """set requires_grad=False"""
        #for param in self.fc1.parameters():
        #    param.requires_grad = False

        #self.fc2 = nn.Linear(self.output_size+step, self.output_size)
        #self.fc3 = nn.Linear(self.input_size+(2*step), self.output_size)

        self.dropout1=  nn.Dropout(p=0.2)
        self.dropout2=  nn.Dropout(p=0.2)
        #self.dropout3=  nn.Dropout(p=0.2)



    def forward(self, x: NetIO) -> NetIO:
        #x = self.dropout1(x)
        #x = self.fc1(x)        
        x = F.relu(self.fc1(x))
        #x = self.dropout2(x)

        x = F.relu(self.fc2(x))
        #x = self.dropout3(x)

        #x = F.relu(self.fc3(x))
        #x = self.dropout2(x)

        return x

class SmallRho(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1, hidden_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        step=(self.output_size - self.input_size)//2
        print("step phi", step)
        print("input_size rho", self.input_size)

        print("output_size rho", self.output_size)
        #self.fc1 = nn.Linear(self.input_size, self.input_size+step)
        #self.fc2 = nn.Linear(self.input_size+step, self.output_size)


        self.fc1 = nn.Linear(self.input_size, self.output_size)
        self.dropout1=  nn.Dropout(p=0.2)
        self.dropout2=  nn.Dropout(p=0.2)
        self.dropout3=  nn.Dropout(p=0.2)
        #self.fc2 = nn.Linear(self.input_size+step, self.output_size)
        #self.fc3 = nn.Linear(self.input_size+(2*step), self.output_size)


    def forward(self, x: NetIO) -> NetIO:
        #x = F.dropout(x)
        #x = self.dropout2(x)        
        x = F.relu(self.fc1(x))

        #x = self.dropout3(x) #good results without this dropout
        #x = F.relu(self.fc1(x))
        #x = self.dropout2(x)

        #x = self.dropout2(x)

        #x = F.relu(self.fc2(x))
        #x = self.dropout3(x)

        #x = F.relu(self.fc3(x))
        #x = self.dropout3(x)
        return x

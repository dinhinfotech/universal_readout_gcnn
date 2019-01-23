from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
import torch.autograd as autograd
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb
from util import cmd_args

sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init

class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        pred = self.h2_weights(h1)

        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            return pred, mae, mse
        else:
            return pred

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout
        print('Input size: ', input_size)

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, acc
        else:
            return logits

class MLP_RNN_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLP_RNN_Classifier, self).__init__()
        hidden_size = input_size
        self.hidden_size = input_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, num_class)
        weights_init(self)
            
    def forward(self, input, y):        
        outputs = []
        for g_idx in range(input.shape[0]):
            hidden = Variable(torch.zeros(1, self.hidden_size))
            for i in range(input.shape[1]):
                combined = torch.cat((input[g_idx][i].view(1,-1), hidden), 1)
                hidden = self.i2h(combined)
                #hidden = F.relu(hidden)
                output = self.i2o(combined)
                output = F.log_softmax(output, dim=1)
            outputs.append(output[0])            

        y = Variable(y)
        outputs = torch.stack(outputs)
        loss = F.nll_loss(outputs, y)
        pred = outputs.data.max(1, keepdim=True)[1]
        acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
        return outputs, loss, acc             

class MLP_LSTM_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(MLP_LSTM_Classifier, self).__init__()
        #hidden_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.hidden2label = nn.Linear(hidden_size, num_class)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_size)))
        
    def forward(self, input, y):
        outputs = []
        for g_idx in range(input.shape[0]):
            lstm_out, self.hidden = self.lstm(input[g_idx].view(cmd_args.sortpooling_k,1,-1), self.hidden)
            #cmd_args.sortpooling_k
            output = self.hidden2label(lstm_out[-1])
            output = F.log_softmax(output)
            outputs.append(output[0])            

        y = Variable(y)
        outputs = torch.stack(outputs)
        loss = F.nll_loss(outputs, y)
        pred = outputs.data.max(1, keepdim=True)[1]
        acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
        return outputs, loss, acc 

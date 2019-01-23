import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from DGCNN_embedding import DGCNN
from DGCNN_deepsets_embedding import DGCNNDeepSets
from DGCNN_rnn_embedding import DGCNN_RNN

from mlp_dropout import MLPClassifier, MLP_RNN_Classifier, MLP_LSTM_Classifier
sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from embedding import EmbedMeanField, EmbedLoopyBP
from util import cmd_args

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        elif cmd_args.gm == 'DGCNN':
            model = DGCNN
        elif cmd_args.gm == 'DGCNN_DS':
            model = DGCNNDeepSets
        elif cmd_args.gm == 'DGCNN_RNN' or 'DGCNN_LSTM':
            model = DGCNN_RNN           
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.gm == 'DGCNN' or cmd_args.gm == 'DGCNN_DS' or 'DGCNN_RNN' or 'DGCNN_LSTM':
            self.s2v = model(latent_dim=cmd_args.latent_dim,
                            output_dim=cmd_args.out_dim,
                            num_node_feats=cmd_args.feat_dim+cmd_args.attr_dim,
                            num_edge_feats=0,
                            k=cmd_args.sortpooling_k)
        else:
            self.s2v = model(latent_dim=cmd_args.latent_dim,
                            output_dim=cmd_args.out_dim,
                            num_node_feats=cmd_args.feat_dim,
                            num_edge_feats=0,
                            max_lv=cmd_args.max_lv)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN' or cmd_args.gm == 'DGCNN_DS':
                out_dim = self.s2v.dense_dim
            else:
                out_dim = cmd_args.latent_dim
        if cmd_args.gm == 'DGCNN' or cmd_args.gm == 'DGCNN_DS':
            self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class, with_dropout=cmd_args.dropout)
        elif cmd_args.gm == 'DGCNN_RNN':
            self.mlp = MLP_RNN_Classifier(input_size=sum(cmd_args.latent_dim), hidden_size=cmd_args.hidden, num_class=cmd_args.num_class, with_dropout=cmd_args.dropout)
        elif cmd_args.gm == 'DGCNN_LSTM':
            self.mlp = MLP_LSTM_Classifier(input_size=sum(cmd_args.latent_dim), hidden_size=cmd_args.hidden, num_class=cmd_args.num_class)
            

    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()

        return node_feat, labels

    def forward(self, batch_graph):
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, None)

        return self.mlp(embed, labels)

def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size, scheduler=None):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    
    if cmd_args.gm == "DGCNN_LSTM":
        retain_graph=True
    else:
        retain_graph=False
        
    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]        
        batch_graph = [g_list[idx] for idx in selected_idx]
        _, loss, acc = classifier(batch_graph)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward(retain_graph=retain_graph)
            optimizer.step()

        loss = loss.data.cpu().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )
        #pbar.set_description((loss, acc))
        total_loss.append( np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples

    #if not classifier.train:
    #    scheduler.step(avg_loss)
    return avg_loss, acc

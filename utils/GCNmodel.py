#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torchvision.models as models
import math
from torch.nn import Parameter
import torch
import utils as utils
import torch.nn as nn
import torch.nn.functional as F
import time
# from GAT import GAT, GraphAttentionLayer, SpGraphAttentionLayer
cuda_gpu = torch.cuda.is_available()
from model.base import BaseModel


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    @staticmethod
    def process_graph(graph_data):
        # graph_data = torch.as_tensor(torch.from_numpy(graph_data), dtype=torch.float32)
        #
        # graph_data =torch.from_numpy(graph_data)
        N = graph_data.shape[0]
        # torch.eye 生成对角线全为1，其余部分都为0的二维数组， get Ab波浪
        matrix_i = torch.eye(N, dtype=graph_data.dtype)
        # graph_data = torch.from_numpy(graph_data)
        # print(graph_data.dtype)
        graph_data += matrix_i

        degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float("inf")] = 0.  # [N]
        # 返回以1D向量为对角线的二D数组
        degree_matrix = torch.diag(degree_matrix)  # n,n
        return torch.mm(degree_matrix, graph_data)  # D^(-1) * A = \hat(A)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNforGRU(nn.Module):
    def __init__(self, in_features, out_features, adj, num_nodes, bias=False):
        super(GCNforGRU, self).__init__()
        self.nodes = num_nodes
        self.adj = adj
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        #self.batch = nn.BatchNorm1d(num_nodes)
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, h):
        graph = torch.tensor(self.adj, dtype=torch.float32)
        graph = GraphConvolution.process_graph(graph)
        ## x (batchsize, node_num, feature)
        #print("self.weight:", self.weight.shape)
        #print("x.shape:", x.shape)
        input = torch.cat((x, h), dim=2)
        #print("input.shape:", input.shape)
        # feature_size = input.shape[2]
        # input = input.permute(1, 2, 0)
        # input = input.reshape(self.nodes, -1)
        #
        # output = torch.matmul(self.adj, input)
        # output = output.view(self.nodes, feature_size, -1)
        # output = output.permute(2, 0, 1)
        # output = output.reshape(output.shape[0]*output.shape[1], -1)
        # #print("self.weight:", self.weight.shape)
        # output = torch.matmul(output, self.weight)
        # if self.bias is not None:
        #     output += self.bias
        # output = output.view(-1, self.nodes, output.shape[1])
        output = torch.matmul(graph, input)  # [N, N], [B, N, ouput] = [B,N,output]
        output = torch.matmul(output, self.weight)  # [B,N,out_put]
        #print("final output.shape:", output.shape)
        #output = self.batch(output)
        return output

class TGCNCell(nn.Module):
    def __init__(self, num_units, _adj, num_nodes):
        super(TGCNCell, self).__init__()
        self.nodes = num_nodes
        self.units = num_units
        #从TGC网络出来时的特征数目
        # 归一化获得邻接矩阵

        # if cuda_gpu:
        #     _adj = _adj.to(device)
        self.gcn_1 = GCNforGRU(2*num_units, 2*self.units, _adj, num_nodes)
        self.gcn_2 = GCNforGRU(2*num_units, self.units, _adj, num_nodes)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # self.A = Parameter(_adj)
        #self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, X, state=None):
        if state is None:
            state = X
        graph_value = self.sigmoid(self.gcn_1(X, state))
        r, z = graph_value.chunk(2, dim=2)
        r_state = r * state
        h_t1 = self.tan(self.gcn_2(X, r_state))
        #源代码中的GRU更新代码是错误的，如下
        new_h = z * state + (1-z) * h_t1
        #print("new_h.shape:", new_h.shape)
        #正确的应该是：
        #new_h = z * h_t1 + (1-z) * state
        return new_h, new_h

class TGCN(BaseModel):
    #seq_len是输入长度，pre_len是预测长度
    def __init__(self, num_units, adj, num_nodes, num_feature,
                  seq_len, pre_len):
        super(TGCN, self).__init__()
        self.seq = seq_len
        self.nodes = num_nodes
        self.tgcncell = TGCNCell(num_units, adj, num_nodes)
        self.seq_linear = nn.Linear(num_feature, num_units)
        self.drop = nn.Dropout(0.2)
        #self.sigmoid = nn.Sigmoid()
        self.pre_linear = nn.Linear(num_units, pre_len)

    def forward(self, x):
        #x = [batchsize, nodes, seq_len, feature]
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[3])
        input = self.drop(self.seq_linear(x))
        #x = [batchsize*nodes*seq_len, feature]
        input = input.view(batch_size, self.nodes, self.seq, input.shape[-1])
        seq_list = []
        for i in range(self.seq):
            if i == 0:
                x, h = self.tgcncell(input[:, :, i, :])
            else:
                x, h = self.tgcncell(input[:, :, i, :], h)
            seq_list.append(x)
        #last_list = [batch_size * nodes, hidden]
        last_list = seq_list[-1]
        # print(last_list.shape)
        output = self.pre_linear(last_list)
        output = output.view(batch_size, self.nodes, -1)
        # print(output.shape)
        return output.unsqueeze(3)


class TGCN_Encoder(nn.Module):
    #seq_len是输入长度，pre_len是预测长度
    def __init__(self, num_units, adj, num_nodes, num_feature,
                  seq_len, device):
        super(TGCN_Encoder, self).__init__()
        self.device = device
        self.seq = seq_len
        self.nodes = num_nodes
        self.tgcncell = TGCNCell(num_units, adj, num_nodes, device)
        self.seq_linear = nn.Linear(num_feature, num_units)
        self.drop = nn.Dropout(0.2)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x = [batchsize, nodes, seq_len, feature]
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[3])
        input = self.drop(self.seq_linear(x))
        #x = [batchsize*nodes*seq_len, feature]
        input = input.view(batch_size, self.nodes, self.seq, input.shape[-1])
        for i in range(self.seq):
            if i == 0:
                x, h = self.tgcncell(input[:, :, i, :])
                seq_list = x.unsqueeze(2)
            else:
                x, h = self.tgcncell(input[:, :, i, :], h)
                seq_list = torch.cat((seq_list, x.unsqueeze(2)), dim=2)
        #output, hidden
        return seq_list, seq_list[:, :, -1, :]

class TGCN_AttnDecoder(nn.Module):
    def __init__(self, num_units, num_nodes, output_size, max_length, device, adj, dropout_p=0.1):
        super(TGCN_AttnDecoder, self).__init__()
        self.nodes = num_nodes
        self.hidden_size = num_units
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.attn = nn.Linear(self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.tgcncell = TGCNCell(num_units, adj, num_nodes, device)
        self.pre_linear = nn.Linear(num_units, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        attn_weight = self.attn.weight
        attn_bias = self.attn.bias
        attn_c_weight = self.attn_combine.weight
        attn_c_bias = self.attn_combine.bias
        torch.nn.init.xavier_uniform_(attn_weight, gain=1)
        torch.nn.init.constant_(attn_bias, 0)
        torch.nn.init.xavier_uniform_(attn_c_weight, gain=1)
        torch.nn.init.constant_(attn_c_bias, 0)

    def forward(self, input, hidden, encoder_outputs):
        # input, hidden [batch, node, units]
        # encoder_outputs [node, batch, units]
        batch = input.shape[0]
        embedded = input

        # hidden [batch, node, units]
        attn_weights = F.softmax(self.attn(hidden), dim=2)
        #attn_w = [batch, node, lens]
        # attn_weights = attn_weights.view(attn_weights.shape[0], -1, 1)
        # print(encoder_outputs.shape)
        attn_applied = torch.bmm(attn_weights.view(attn_weights.shape[0] * attn_weights.shape[1], 1, attn_weights.shape[2]),
                                 encoder_outputs.view(encoder_outputs.shape[0] * encoder_outputs.shape[1], self.max_length, self.hidden_size))
        # print(attn_applied.shape)
        attn_applied = attn_applied.view(batch, self.nodes, attn_applied.shape[2])
        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)
        # print('output: ', output.shape)
        output = F.relu(output)
        #print('output: ', output.shape)
        output, hidden = self.tgcncell(output, hidden)
        output = self.pre_linear(output)
        return output, hidden, attn_weights

class EncoderDecoderAtt(nn.Module):
    def __init__(self, encoder, decoder, time_step, **kwargs):
        super(EncoderDecoderAtt, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.step = time_step
    def forward(self, X, *args):
        encoder_outputs, hidden = self.encoder(X)
        #print('encoder_outputs: ', encoder_outputs.shape)
        #outputs, hidden, attn_weights = self.decoder(hidden, hidden, encoder_outputs)
        attn_weights = []
        outputs = []
        for i in range(self.step):
            output, hidden, attn_weight = self.decoder(hidden, hidden, encoder_outputs)
            attn_weights.append(attn_weight)
            if i == 0:
                outputs = output
            else:
                outputs = torch.cat((outputs, output), 2)
        return outputs, attn_weights

class ResTGCN(nn.Module):
    def __init__(self, num_units, adj, num_nodes, num_feature,
                  seq_len, pre_len, device):
        super(ResTGCN, self).__init__()
        self.pre= pre_len
        self.device = device
        self.seq = seq_len
        self.nodes = num_nodes
        self.units = num_units
        self.tgcncell = TGCNCell(num_units, adj, num_nodes, device)
        self.seq_linear = nn.Linear(num_feature, num_units)
        self.drop = nn.Dropout(0.2)

        self.attn = nn.Linear(num_units, seq_len)
        self.attn_combine = nn.Linear(num_units, num_units)
        self.pre_linear = nn.Linear(num_units, pre_len)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = [batchsize, nodes, seq_len, feature]
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[3])
        input = self.drop(self.seq_linear(x))
        # x = [batchsize*nodes*seq_len, feature]
        input = input.view(batch_size, self.nodes, self.seq, input.shape[-1])
        for i in range(self.seq):
            if i == 0:
                x, h = self.tgcncell(input[:, :, i, :])
                seq_list = x.unsqueeze(2)
            else:
                x, h = self.tgcncell(input[:, :, i, :], h)
                seq_list = torch.cat((seq_list, x.unsqueeze(2)), dim=2)
        output = seq_list[:, :, -1, :]
        #print(seq_list.shape)
        attn_weights = F.softmax(self.attn(output), dim=2)
        attn_applied = torch.bmm(
            attn_weights.view(attn_weights.shape[0] * attn_weights.shape[1], 1, attn_weights.shape[2]),
            seq_list.view(seq_list.shape[0] * seq_list.shape[1], self.seq, self.units))
        attn_applied = attn_applied.view(batch_size, self.nodes, attn_applied.shape[2])
        #res
        output = attn_applied
        output = self.pre_linear(output)
        return output, attn_weights

# if __name__ == "__main__":
    # batchsize, nodes, seq, feature
    # batchsize, nodes, seq, feature


    #T_GCN = TGCN(num_units=64, adj=adj, num_nodes=184, num_feature=8,
    #            seq_len=16, pre_len=8, device=device)
    #y_hat = T_GCN(example)
    #print(y_hat.shape)

    # print(y_hat.shape)
    #
    # Encoder = TGCN_Encoder(num_units=64, adj=adj, num_nodes=node, num_feature=8,
    #                                  seq_len=16, device=device)
    # Decoder = TGCN_AttnDecoder(num_units=64, num_nodes=node, output_size=1, max_length=16, device=device, adj=adj)
    # ED = EncoderDecoderAtt(encoder=Encoder, decoder=Decoder, time_step=8)
    # outputs, attn = ED(example)
    # print(outputs.shape)
    # print(len(attn), ": ", attn[0].shape)




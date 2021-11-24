import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from model.TCN import TemporalConvNet,Inception_Temporal_Layer


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''

    def __init__(self, nodes,features,timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.w1 = Parameter(torch.Tensor(timesteps, 1))
        self.w2 = Parameter(torch.Tensor(features, timesteps))
        self.w3 = Parameter(torch.Tensor(features, 1))
        self.e1 = Parameter(torch.Tensor(1, nodes, nodes))
        self.e2 = Parameter(torch.Tensor(nodes, nodes))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w1, gain=1)
        torch.nn.init.xavier_uniform_(self.w2, gain=1)
        torch.nn.init.xavier_uniform_(self.w3, gain=1)
        torch.nn.init.xavier_uniform_(self.e1, gain=1)
        torch.nn.init.xavier_uniform_(self.e2, gain=1)

    def forward(self, X):
        '''
        input:[B,N,T,D] 此处用于周数据的处理，所以T=7
        :return: [B,T,T]
        '''
        # out1 [B,node,D]
        # print(X.shape)
        # print(self.w1.shape)
        out1 = torch.matmul(X.permute(0, 1, 3, 2), self.w1).squeeze(3)
        # out2 [B,node,t]
        out2 = torch.matmul(out1, self.w2)
        # out3 [B,t,node]
        out3 = torch.matmul(X.permute(0, 2, 1, 3), self.w3).squeeze(3)

        out = torch.matmul(out2, out3)

        e = torch.matmul(self.e2, torch.sigmoid(out + self.e1))
        e = e - torch.max(e, dim=1, keepdim=True).values

        exp = torch.exp(e)
        e_norm = exp / torch.sum(exp, dim=1, keepdim=True)

        return e_norm

class Time_Attention(nn.Module):
    def __init__(self, nodes,features,timesteps):
        super(Time_Attention, self).__init__()
        self.w1 = Parameter(torch.Tensor(nodes,1))
        self.w2 = Parameter(torch.Tensor(features,nodes))
        self.w3 = Parameter(torch.Tensor(features, 1))
        self.e1 = Parameter(torch.Tensor(1,timesteps, timesteps))
        self.e2 = Parameter(torch.Tensor( timesteps, timesteps))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w1, gain=1)
        torch.nn.init.xavier_uniform_(self.w2, gain=1)
        torch.nn.init.xavier_uniform_(self.w3, gain=1)
        torch.nn.init.xavier_uniform_(self.e1, gain=1)
        torch.nn.init.xavier_uniform_(self.e2, gain=1)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.w1.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)

    def forward(self,X):
        '''
        input:[B,N,pre,T] 此处用于周数据的处理，所以T=7
        :return: [B,T,T]
        '''
        #out2 [B,T,node]
        out1 = torch.matmul(X.permute(0,3,2,1),self.w1).squeeze(3)

        out2 = torch.matmul(out1,self.w2)
        out3 = torch.matmul(X.permute(0,1,3,2),self.w3).squeeze(3)
        out = torch.matmul(out2,out3)

        e = torch.matmul(self.e2,torch.sigmoid(out + self.e1))
        e = e - torch.max(e,dim=1,keepdim=True).values

        exp = torch.exp(e)
        e_norm = exp/torch.sum(exp,dim=1,keepdim=True)

        return e_norm

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        input:[B,D,N,T]
        output:[B,D,N,T]
        :param x:
        :return:
        '''
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

     
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

    
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
#        print(y.shape)
        return x * y.expand_as(x)

class BasicConv2D(nn.Module):
    def __init__(self,c_in,c_out,kernel_size,stride=1,padding = (0,0)):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=c_in,out_channels=c_out,kernel_size=(1,kernel_size),stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(c_out,eps=0.001)

    def forward(self,x):
        '''
        :param x: [B,c_in,N,T]
        :return: [B,c_out,N,T]
        '''
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class TimeBlock(nn.Module):
    def __init__(self,in_c,out_c,T,t_re):
        '''
        :param in_c:
        :param out_c:
        :param t_re: T要减少的值
        '''
        super(TimeBlock, self).__init__()
        self.line1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3),padding=(0,1),stride=1),
            BasicConv2D(c_in=in_c,c_out=int(out_c/4),kernel_size=1))
        self.line2 = BasicConv2D(c_in=in_c,c_out=int(out_c/2),kernel_size=3,padding=(0,1))
        self.line3 = BasicConv2D(c_in=in_c,c_out=int(out_c/4),kernel_size= 5,padding=(0,2))
        self.line4 = BasicConv2D(c_in=in_c,c_out=int(out_c/4),kernel_size=1)
        self.conv_att = eca_layer()
        self.linear = nn.Linear(T, T - t_re)

    def forward(self,x):
        '''
        :param x: [B,N,T,c_in]
        :return:  [B,N,T,c_out]
        '''
        x0 = x.permute(0,3,1,2)
        # x1 = self.line1(x0)
        x2 = self.line2(x0)
        x3 = self.line3(x0)
        x4 = self.line4(x0)
        # print('start',x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        in_out = torch.cat([x2,x3,x4],dim=1)
        out = F.relu(self.conv_att(in_out))
        out = self.linear(out)
        return out.permute(0, 2, 3, 1)


class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x

class CNNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=5):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(CNNBlock, self).__init__()
        self.k = kernel_size
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_att = eca_layer()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.align = align(in_channels, out_channels)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        T = X.shape[2]
        temp = self.align(X.permute(0, 3, 2, 1))[:, :, self.k-1:, :]
        temp =temp.permute(0,1,3,2)
        X = X.permute(0, 3, 1, 2)

        # print("temp",temp.shape)
        # print((self.conv1(X) + temp).shape)
        # print((self.conv1(X)).shape)


        #这边有个残差机制，我想在这里尝试inception+res
        out = (self.conv1(X) + temp) * torch.sigmoid(self.conv2(X))
        # out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = F.relu(self.conv_att(out))
        out = out.permute(0, 2, 3, 1)
        return out

class graphblock(nn.Module):
    def __init__(self,input_c,output_c):
        super(graphblock, self).__init__()
        self.out_c = output_c
        self.in_c =input_c
        self.line1 = nn.Linear(in_features=input_c,out_features=2*output_c)

    def forward(self,x,graph):
        '''
        :param x: 【B,n,T,D】
        :param graph: [n,n]
        :return: his_pre, future_pre
        '''
        # print(self.in_c)
        # print(x.shape[3])
        if(graph.shape[0] == graph.shape[1]):
            out = torch.einsum("jj,jklm->kjlm", graph, x.permute(1, 0, 2, 3))
        else:
            out = torch.einsum("kjj,kjlm->kjlm", graph, x)
        out1 = F.relu(self.line1(out))
        return x-out1[:,:,:,:self.out_c],out1[:,:,:,self.out_c:]

class MutiGraph(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(MutiGraph, self).__init__()
        self.muGcn = nn.Sequential()
        self.muGcn.add_module('gcn1',graphblock(in_channels,out_channels))
        self.muGcn.add_module('gcn2',graphblock(in_channels,out_channels))
        self.muGcn.add_module('gcn3',graphblock(in_channels,out_channels))

    def forward(self,x,graphs,sat):
        grapu_n = graphs.shape[0]
            #此时input【B,n,T,D】
        graph = graphs[0]*sat
        his, out = self.muGcn[0](x, graph)
        for i in range(1,grapu_n):
            his,out1 = self.muGcn[i](his, graphs[i])
            out = torch.cat([out,out1],dim=3)

        return out

class STGMNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """
    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes,gc,diffusion_step,T,kt=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGMNBlock, self).__init__()
        # self.temporal1 = CNNBlock(in_channels=in_channels,out_channels=out_channels)
        self.temporal1 = TimeBlock(in_c=in_channels,out_c=out_channels,T=T,t_re=4)
        self.SAt = Spatial_Attention_layer(nodes=num_nodes, features=out_channels, timesteps=T-4)
        self.gc = gc
        # self.Timeatt = Time_Attention(nodes=num_nodes, features=in_channels, timesteps=12)
        self._max_diffusion_step = diffusion_step

        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels * kt,
                                                     out_channels))

        # self.temporal2 = CNNBlock(in_channels=spatial_channels, out_channels=out_channels)
        # self.temporal2 = TimeBlock(in_c=in_channels, out_c=out_channels, T=8, t_re=4)
        self.linear = nn.Linear(spatial_channels,out_channels)
        self.line2 =nn.Linear(2*out_channels,out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()
        self.gcn = MutiGraph(out_channels,out_channels)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, adj):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """

        #tat [B,N,D,T]
        # week_in_tat = self.Timeatt(X.permute(0,1,3,2))
        # week_in = torch.einsum("ijmk,ikk->ijkm", X.permute(0,1,3,2), week_in_tat)

        t = self.temporal1(X)
        B, N, T, D = t.shape

        spatial_At = self.SAt(t)
        out = self.gcn(t,adj,spatial_At)

        t2 = F.relu(torch.matmul(out, self.Theta1))
        temp = torch.cat([t2,t],dim=3)
        r = F.sigmoid(self.line2(temp))
        t3 = (1-r)*t2+r*t
        # t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3

class STGMNBlock_week(nn.Module):
    def __init__(self,num_nodes,pre_len,timesteps,gc, kt=3,diffusion_step =3,
                 ):
        super(STGMNBlock_week, self).__init__()
        self.Timeatt = Time_Attention(nodes=num_nodes, features=pre_len, timesteps=timesteps)
        self.linear1 = nn.Linear(timesteps, 64)
        self.gc = gc
        if(self.gc == 'dgcn'):
            self.linear2 = nn.Linear((diffusion_step+1)*pre_len*timesteps,64)
        else:
            self.linear2 = nn.Linear(pre_len*kt*timesteps,64)

        self._max_diffusion_step = diffusion_step

    def forward(self,X,adj):
        '''
        Input:[B,N,pre,T]
        output [b,n,t,output]
        :param X:
        :return:
        '''
        week_in_tat = self.Timeatt(X)
        # [b,t,t] *[B,N,pre,T]
        # print(X.shape)
        # print(week_in_tat.shape)
        # week_in = torch.matmul(X[:,:,:7,:].permute(0,1,3,2),week_in_tat)
        week_in = torch.einsum("ijmk,ikk->ijmk", X, week_in_tat)

        # week_in = week_in.permute(0,1,3,2)
        B,N,pre,T = week_in.shape

        temp = torch.unsqueeze(X,0)
        x0 = X.clone()
        if(self.gc == "dgcn"):
            for _adj in adj:
                x1 = torch.einsum("jj,bjlm->bjlm", _adj, x0)
                # x1 = torch.matmul(x0,_adj)
                temp = STGMNBlock._concat(temp,x1)
                for k in range(2,self._max_diffusion_step +1):
                    x2 = 2*torch.einsum("jj,bjlm->bjlm", _adj, x1) - x0
                    temp  = STGMNBlock._concat(temp,x2)
                    x1,x0 = x2,x1

            num_m = len(adj)*self._max_diffusion_step +1
            lfs  = temp.reshape(B,N,pre,num_m,T)
            lfs = lfs.view(B,N,pre*num_m*T)
        else:
            lfs = torch.einsum("tjj,jklm->kjltm", adj, week_in.permute(1, 0, 2, 3))
            B, N, pre, kt, T = lfs.shape[0], lfs.shape[1], lfs.shape[2], lfs.shape[3], lfs.shape[4]
            # print(lfs.shape)
            lfs = lfs.contiguous().view(B, N, pre * kt * T)
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))

        # print(lfs.shape)
        t2 = F.relu(self.linear2(lfs))

        return t2.unsqueeze(2)


class STGMN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output,graph,gc,diffusion_step):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGMN, self).__init__()
        if(gc == "chebi"):
            self.kt = 3
        if(gc == 'gcn'or gc == 'dgcn'):
            self.kt = 1
        self.gc = gc
        self.block_w = STGMNBlock_week(pre_len = num_timesteps_output,num_nodes=num_nodes,timesteps=7,kt=self.kt,gc= self.gc,diffusion_step = diffusion_step)
        self.block1 = STGMNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes,kt=self.kt,gc= self.gc,diffusion_step = diffusion_step,T=12)
        self.block2 = STGMNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes,kt=self.kt,gc= self.gc,diffusion_step = diffusion_step,T=8)
        self.last_temporal = CNNBlock(in_channels=64, out_channels=64,kernel_size=3)
        self.fully = nn.Linear((num_timesteps_input- 2 * 5) * 64,
                               num_timesteps_output)
        self.graph = graph
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(torch.device("cuda")), requires_grad=True).to(torch.device("cuda"))
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(torch.device("cuda")), requires_grad=True).to(torch.device("cuda"))

    def forward(self, X,device):
        """
        :param X: Input data of shape (B, num_nodes, seq,D).
        :param A_hat: Normalized adjacency matrix.
        """
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        graph = torch.cat([self.graph.to(device),adp.unsqueeze(0)],dim=0)
        #zengjiale 了一个自适应

        X = X.to(device)
        # graph = graph.to(device)

        # print("dddd",X.shape)  (B, num_nodes, seq,D+7)

        # out_w = self.block_w(X[:, :, :, 3:],graph)
        # print(out_w.shape)
        # (B, num_nodes, pre , 7) 输入数据分割 预测未来12个值，每个值都对应过去7个his
        #对于过去一个小时的数据、用因果卷积获得其中的
        out1 = self.block1(X[:,:,:,:3], graph)
        # print(out1.shape) (B, num_nodes, seq-(kt+1)*2 , nuits)
        out2 = self.block2(out1, graph)
        # print(out2.shape) (B, num_nodes, seq-(kt+1)*4 , nuits)
        out3 = self.last_temporal(out2)
        # print("out3",out3.shape) (B, num_nodes, seq-(kt+1)*5 , nuits)
        # print(out3.shape)
        # 暂时不算上多出来的out3 = torch.cat([out3,out_w],dim=2)
        # print(out3.shape)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))

        return out4.unsqueeze(3)



# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_processing import *
from utils.graph import get_adjacent_matrix,process_graph,get_adjacent_matrix_2,calculate_dgcn
from utils.metrics import m_MAE,m_MSE,m_RMSE
from model.TGCN import *
from model.STGCN import *
from model.STGMN import *
# from model.GCNmodel import TGCN
# from GCNmodels import *
import numpy

"""
     Dataset description：
     PeMS04 ，加利福尼亚高速数据，"data.npz"。
"""


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True

def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)


def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    '''
    参数设置
    '''
    dataset ="PEMS08"  #数据集
    history_len = 12   #历史数据
    pre_len = 12     #预测数据长度
    day_split = True   #划分训练、测试集合
    gcn_choose = 'chebi'  #"gcn","chebi","dgcn","Ada"
    diffusion_step = 1

    if(dataset == "PEMS04"):
        if torch.cuda.is_available():
            adj_file = "/home/ZhangM/expGCN/data/PEMS04/distance.csv"
        else:
            adj_file = "data/PEMS04/distance.csv"
        num_nodes = 307
    elif(dataset == "PEMS08"):
        if torch.cuda.is_available():
            adj_file = "/home/ZhangM/expGCN/data/PEMS08/distance.csv"
        else:
            adj_file = "data/PEMS08/distance.csv"
        num_nodes = 170

    # Loading Dataset
    train_loader, test_loader, scaler,train_data,test_data = get_dataloader(dataset =dataset,day_split=day_split,time_interval=5,
                                                                               seq_len=history_len,pre_len=pre_len,batch_size=50)
    #Load adj 
    adj = get_adjacent_matrix(adj_file, num_nodes=num_nodes)
    # print(adj)
    # adj = torch.tensor(adj, dtype=torch.float32)
    if(gcn_choose == "chebi"): #[kt,n,n]
        adj = scaled_laplacian(adj)
        print(adj)
        graph = cheb_poly(adj, 2)
        graph = torch.Tensor(graph.astype(np.float32))
    if(gcn_choose == "gcn"): #[1,n,n]
        graph = process_graph(adj)
        graph = graph.unsqueeze(0)
    if(gcn_choose == "dgcn"):
        graph = calculate_dgcn(adj)
        # print(graph)
        graph = graph.unsqueeze(0)

    # adj = process_graph(adj)
    print(graph)


    # print(graph)

    # Loading Model
    # model = GCN(input=12, hidden=32 ,output=12)
    # model = TGCN(num_units=32, num_feature=3,seq_len=history_len, pre_len=pre_len, adj=graph)
    # model = GRU(input_dim=3, hidden_dim=32, output_dim=pre_len,layers=1)
    #mygcn使用connect的chebi 2阶
    model = STGMN(num_nodes = num_nodes, num_features=3, num_timesteps_input = history_len,
                 num_timesteps_output = pre_len, graph=graph,gc = gcn_choose,diffusion_step = diffusion_step)

    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(),lr = 0.0005)

    # Train model
    Epoch = 50

    model.train()
    for epoch in range(Epoch):
        epoch_loss = 0.0
        start_time = time.time()
        for data,target in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            model.zero_grad()
            # print(np.shape(data))
            predict_value = model(data, device).to(torch.device("cuda"))  # [0, 1] -> recover
            # predict_value = model(data).to(torch.device("cpu"))

            # print(target.shape)
            loss = criterion(predict_value, target)
            # print(predict_value.shape)
            # print(data["flow_y"].shape)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        end_time = time.time()

        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, epoch_loss / len(train_loader),
                                                                          (end_time-start_time)/60))

    # Test Model
    model.eval()
    with torch.no_grad():
        pred = []
        y = []
        total_loss = 0.0
        for data,target in test_loader:
            predict_value = model(data, device).to(torch.device("cuda"))  # [B, N, 1, D]
            # predict_value = model(data)
            loss = criterion(scaler.inverse_transform(predict_value), scaler.inverse_transform(target))
            # print(predict_value)
            pred.append(scaler.inverse_transform(predict_value))
            y.append(scaler.inverse_transform(target))
            # loss = criterion(predict_value,data["flow_y"])

            # print(LoadData.recover_data(data["norm"][0], data["norm"][1],predict_value))
            # print(LoadData.recover_data(data["norm"][0],data["norm"][1],data["flow_y"]))
            total_loss += loss.item()

        print(torch.cat(pred,dim=0).shape)
        print("mae:{:02.4f}, mse:{:02.4f},rmse{:02.4f}".format(m_MAE(torch.cat(pred,dim=0),torch.cat(y,dim=0)),m_MSE(torch.cat(pred,dim=0),torch.cat(y,dim=0)),m_RMSE(torch.cat(pred,dim=0),torch.cat(y,dim=0))))
        print("Test Loss: {:02.4f}".format( total_loss / len(test_loader)))


if __name__ == '__main__':
    seed = 10
    setup_seed(seed)
    main()

import numpy as np
import os
from  utils.Norm import MinMax01Scaler,StandardScaler
import torch
from utils import get_timef
import datetime




def load_data(dataset):
    """
    :param flow_file: str,交通流量数据的 .npz 文件路径
    :return: np.array(N, T, D)
    """
    if dataset == 'PEMS04':
        # data_path = os.path.join('data/PEMS04/pems04.npz')
        data = np.load('../data/PEMS04/pems04.npz')['data']
        # data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMS08':
        # data_path = os.path.join('/home/ZhangM/expGCN/data/PEMS08/pems08.npz')
        data_path = os.path.join('../data/PEMS08/pems08.npz')
        data = np.load(data_path)['data']
        # data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    else:
        raise ValueError
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    # tc = get_timef.TimeCovariates(datetime.datetime(2016, 7, 1), num_ts=17856, freq="5T", normalized=False)
    # vars = tc.get_covariates()
    # vars = np.expand_dims(vars.transpose(1, 0), 2)
    # one = np.ones((1, 170))
    # vars = np.dot(vars, one).transpose(0, 2, 1)
    # data = np.concatenate([data,vars],axis=2)
    # print(data.shape)
    # return np.expand_dims(data,2)
    return data

def normalize_dataset(data,normalizer):
    if normalizer == 'max01':
        minimum = data.min()
        maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'std':
        mean = data.mean()
        std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')

    return data,scaler

def split_data1(data,trian,time_interval):
    T = int(24 * 60 / time_interval)
    train_data= data[:int(trian*T)]
    test_data = data[int((T*(trian))):]
    print("train",train_data.shape)
    print("test", test_data.shape)
    return train_data, test_data

def split_data2(data,trian):
    T = data.shape[0]
    train_data= data[:int(T*trian)]
    test_data = data[int((T*(trian))):]
    return train_data,test_data

def get_XY(data,seq_len,pre_len,single=False):
    '''
        :param data: shape [B, ...]
        :param window:
        :param horizon:
        :return: X is [B, W, ...], Y is [B, H, ...]
        '''
    length = len(data)
    end_index = length - seq_len - pre_len + 1
    X = []  # windows
    Y = []  # horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index + seq_len])
            Y.append(data[index + seq_len + pre_len - 1:index + seq_len+ pre_len][:,:,0])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index + seq_len])
            Y.append(data[index + seq_len:index + seq_len + pre_len][:,:,0])
            index = index + 1
    # X = np.expand_dims(X, 3)
    Y = np.expand_dims(Y,3)
    # print("d",np.array(Y).shape)
    X = np.array(X).transpose(0,2,1,3)  #[B,N,len,D]
    Y = np.array(Y).transpose(0,2,1,3)
    # X= np.swapaxes(X,dim0=1,dim1=2)
    return X, Y

def get_XY_his(data,seq_len,pre_len,single=False):
    '''
        对于输入参数的选择，不仅选择临近的历史数据，还选择过去《前 7天》（数据量也不太够、所以选的不多。。），每一天同一时刻的数据
        :param data: shape [B, ...]
        :param window:
        :param horizon:
        :return: X is [B, W, ...], Y is [B, H, ...]
        '''
    length = len(data)
    end_index = length - seq_len - pre_len + 1
    X = []  # windows
    Y = []  # horizon
    x_w = []
    index = 288*7 - seq_len #一天的数据量是12*24 = 288
    if single:
        while index < end_index:
            X.append(data[index:index + seq_len])
            Y.append(data[index + seq_len + pre_len - 1:index + seq_len+ pre_len][:,:,0])
            index = index + 1
    else:
        while index < end_index:
            x_h = []
            for j in range(pre_len):
                day_his = data[index - 288*7 + seq_len +j : index :288][:,:,0]
                x_h.append(day_his)
            x_w.append(x_h)
            X.append(data[index:index + seq_len])
            # X.append(day_his + data[index:index + seq_len])
            Y.append(data[index + seq_len:index + seq_len + pre_len][:,:,0])
            index = index + 1
    # X = np.expand_dims(X, 3)
    Y = np.expand_dims(Y,3)
    x_w = np.array(x_w).transpose(0,3,1,2)

    X = np.array(X).transpose(0,2,1,3)  #[B,N,len,D]
    Y = np.array(Y).transpose(0,2,1,3)
    X = np.concatenate((X,x_w),axis=3)
    return X, Y



def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X,Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def get_dataloader(dataset,normalizer='std',day_split= True,time_interval=5,single = False,
                   seq_len=12,pre_len=12,batch_size=64, sameday_his = True):
    data = load_data(dataset)
    data, scaler = normalize_dataset(data, normalizer)
    if(day_split):
        data_train, data_test = split_data1(data, trian=50, time_interval=time_interval)
    else:
        data_train,  data_test = split_data2(data,trian=0.7)

    if(sameday_his == False):
        x_tra, y_tra = get_XY(data_train, seq_len, pre_len, single)
        x_test, y_test = get_XY(data_test, seq_len, pre_len, single)
    else:
        x_tra, y_tra = get_XY_his(data_train, seq_len, pre_len, single)
        x_test, y_test = get_XY_his(data_test, seq_len, pre_len, single)

    print('Train: ', x_tra.shape, y_tra.shape)
    print('Test: ', x_test.shape, y_test.shape)
    train_dataloader = data_loader(x_tra,y_tra,batch_size, shuffle=True, drop_last=True)
    test_dataloader = data_loader(x_test,y_test, batch_size, shuffle=False, drop_last=False)
    return train_dataloader, test_dataloader, scaler,x_tra.shape[0],x_test.shape[0]

if __name__ == '__main__':
    get_dataloader("PEMS08")
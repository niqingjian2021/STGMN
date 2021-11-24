# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy.linalg as la
import math
from sklearn.svm import SVR
from statsmodels.tsa.arima_model import ARIMA
from utils.data_processing import load_data

def preprocess_data(data, time_len, rate, seq_len, pre_len):
    data1 = np.mat(data)
    T = int(24 * 60 / 5)
    train_data = data1[:int(50 * T)]
    test_data = data1[int((T * (50))):]
    # train_size = int(time_len * rate)
    # train_data = data1[0:train_size]
    # test_data = data1[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])
    return trainX, trainY, testX, testY


###### evaluation ######
def evaluation(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a - b) / la.norm(a)
    r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    var = 1 - (np.var(a - b)) / np.var(a)
    return rmse, mae, 1 - F_norm, r2, var



data = load_data('PEMS08')[:, :, 0]
time_len = data.shape[0]
num_nodes = data.shape[1]
train_rate = 0.8
seq_len = 12
pre_len = 12
trainX, trainY, testX, testY = preprocess_data(data, time_len, train_rate, seq_len, pre_len)
print(np.shape(trainX))
print(np.shape(trainY))
print(np.shape(testX))
print(np.shape(testY))
method = 'HA'  ####HA or SVR or ARIMA

########### HA #############
if method == 'HA':
        result = []
        real = []
        result1=[]
        testY = np.array(testY)
        rmse, mae, acc, r2, var, pred, ori = [], [], [], [], [], [], []
        result_mae = []
        result_mse = []
        for j in range(12):
            result = []
            print(j)
            for i in range(len(testX)):
                a = testX[i]
                a1 = np.mean(a, axis=0)
                #            print(np.shape(a1))
                result.append(a1)
            if (np.ndim(result) == 2):
                result = np.expand_dims(result, 1)

            testX = np.array(testX)[:, 1:, :]
            testX = np.concatenate((result, testX), axis=1)
            result = np.reshape(np.array(result), [-1, num_nodes])
            result1.append(result)

            er_rmse, er_mae, er_acc, r2_score, var_score = evaluation(result, testY[:,j,:])
            print(er_mae)
            result_mae.append(er_mae)
            print(er_rmse)
            result_mse.append(er_rmse)
            rmse.append(er_rmse)
            mae.append(er_mae)
            acc.append(er_acc)
            r2.append(r2_score)
            var.append(var_score)

        file1 = open('HA_mae_8.txt','w')
        file1.write(str(result_mae))
        file1.close()
        file2 = open('HA_rmse_8.txt', 'w')
        file2.write(str(result_mse))
        file2.close()
        print('arima_rmse:%r' % (np.mean(rmse)),
              'arima_mae:%r' % (np.mean(mae)),
              # 'arima_acc:%r' % (np.mean(acc1)),
              'arima_r2:%r' % (np.mean(r2)),
              'arima_var:%r' % (np.mean(var)))

############ SVR #############
if method == 'SVR':
    total_rmse, total_mae, total_acc, result = [], [], [], []
    for i in range(num_nodes):
        data1 = np.mat(data[:100])
        a = data1[:, i]
        a_X, a_Y, t_X, t_Y = preprocess_data(a, time_len, train_rate, seq_len, pre_len)
        a_X = np.array(a_X)
        a_X = np.reshape(a_X, [-1, seq_len])
        a_Y = np.array(a_Y)
        a_Y = np.reshape(a_Y, [-1, pre_len])
        a_Y = np.mean(a_Y, axis=1)
        t_X = np.array(t_X)
        t_X = np.reshape(t_X, [-1, seq_len])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y, [-1, pre_len])
        print(a_X.shape)
        svr_model = SVR(kernel='linear')
        svr_model.fit(a_X, a_Y)
        pre = svr_model.predict(t_X)
        pre = np.array(np.transpose(np.mat(pre)))
        pre = pre.repeat(pre_len, axis=1)
        result.append(pre)
        print(1)
    result1 = np.array(result)

######## ARIMA #########
if method == 'ARIMA':
    rng = pd.date_range('1/3/2012', periods=16992, freq='5min')
    a1 = pd.DatetimeIndex(rng)
    data = pd.DataFrame(data)
    data.index = a1
    num = data.shape[1]
    rmse, mae, acc, r2, var, pred, ori = [], [], [], [], [], [], []
    for i in range(156):
        ts = data.iloc[:, i]
        ts_log = np.log(ts)
        ts_log = np.array(ts_log, dtype=np.float)
        where_are_inf = np.isinf(ts_log)
        ts_log[where_are_inf] = 0
        ts_log = pd.Series(ts_log)
        ts_log.index = a1
        model = ARIMA(ts_log, order=[1, 0, 0])
        properModel = model.fit()
        predict_ts = properModel.predict(4, dynamic=True)
        log_recover = np.exp(predict_ts)
        ts = ts[log_recover.index]
        er_rmse, er_mae, er_acc, r2_score, var_score = evaluation(ts, log_recover)
        rmse.append(er_rmse)
        mae.append(er_mae)
        acc.append(er_acc)
        r2.append(r2_score)
        var.append(var_score)
    #    for i in range(109,num):
    #        ts = data.iloc[:,i]
    #        ts_log=np.log(ts)
    #        ts_log=np.array(ts_log,dtype=np.float)
    #        where_are_inf = np.isinf(ts_log)
    #        ts_log[where_are_inf] = 0
    #        ts_log = pd.Series(ts_log)
    #        ts_log.index = a1
    #        model = ARIMA(ts_log,order=[1,1,1])
    #        properModel = model.fit(disp=-1, method='css')
    #        predict_ts = properModel.predict(2, dynamic=True)
    #        log_recover = np.exp(predict_ts)
    #        ts = ts[log_recover.index]
    #        er_rmse,er_mae,er_acc,r2_score,var_score = evaluation(ts,log_recover)
    #        rmse.append(er_rmse)
    #        mae.append(er_mae)
    #        acc.append(er_acc)
    #        r2.append(r2_score)
    #        var.append(var_score)
    acc1 = np.mat(acc)
    acc1[acc1 < 0] = 0
    print('arima_rmse:%r' % (np.mean(rmse)),
          'arima_mae:%r' % (np.mean(mae)),
          'arima_acc:%r' % (np.mean(acc1)),
          'arima_r2:%r' % (np.mean(r2)),
          'arima_var:%r' % (np.mean(var)))


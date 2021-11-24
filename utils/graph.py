import numpy as np
import torch
import csv
import scipy.sparse as sp

def get_adjacent_matrix(distance_file: str, num_nodes: int,  graph_type="connect") -> np.array:
    """
    :param distance_file: str, 用于保存节点之间距离的文件
    :param num_nodes: int, number of nodes in the graph
    :param id_file: str, 保存节点之间绝对顺序的文件.
    :param graph_type: str, ["connect", "distance"] ，可以选择是否使用距离作为边的权重
    :return:  邻接矩阵A ，np.array(N, N)
    """
    A = np.zeros([int(num_nodes), int(num_nodes)])
    with open(distance_file, "r") as f_d:
        f_d.readline()
        reader = csv.reader(f_d)
        for item in reader:
            if len(item) != 3:
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])

            if graph_type == "connect":
                A[i, j], A[j, i] = 1., 1.
            elif graph_type == "distance":
                A[i, j] = 1. / distance
                A[j, i] = 1. / distance
            else:
                raise ValueError("graph type is not correct (connect or distance)")
    return A

def get_adjacent_matrix_danxiang(distance_file: str, num_nodes: int,  graph_type="connect") -> np.array:
    """
    :param distance_file: str, 用于保存节点之间距离的文件
    :param num_nodes: int, number of nodes in the graph
    :param id_file: str, 保存节点之间绝对顺序的文件.
    :param graph_type: str, ["connect", "distance"] ，可以选择是否使用距离作为边的权重
    :return:  邻接矩阵A ，np.array(N, N)
    """
    A = np.zeros([int(num_nodes), int(num_nodes)])
    with open(distance_file, "r") as f_d:
        f_d.readline()
        reader = csv.reader(f_d)
        for item in reader:
            if len(item) != 3:
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])

            if graph_type == "connect":
                # A[i, j], A[j, i] = 1., 1.
                A[i, j] = 1.
            elif graph_type == "distance":
                A[i, j] = 1. / distance
                A[j, i] = 1. / distance
            else:
                raise ValueError("graph type is not correct (connect or distance)")
    return A

def get_adjacent_matrix_2(distance_file: str, num_nodes: int,  graph_type="connect") -> np.array:
    A = np.zeros([int(num_nodes), int(num_nodes)])
    kkk= 0
    with open(distance_file, "r") as f_d:
        f_d.readline()
        reader = csv.reader(f_d)
        for item in reader:
            if len(item) != 3:
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])

            if graph_type == "connect":
                distance =distance/10000
                w = np.exp(-distance*distance/0.1)
                # print(distance)
                if w >= 0.5:
                    A[i, j], A[j, i] = w,w
                    kkk = kkk+1
                else:
                    A[i, j], A[j, i] = 0.,0.
                # A[i, j], A[j, i] = 1., 1.
            elif graph_type == "distance":
                A[i, j] = 1. / distance
                A[j, i] = 1. / distance
            else:
                raise ValueError("graph type is not correct (connect or distance)")
    print(kkk)
    return A

def process_graph(graph_data):
        graph_data = torch.as_tensor(torch.from_numpy(graph_data), dtype=torch.float32)
        N = graph_data.shape[0]
        #torch.eye 生成对角线全为1，其余部分都为0的二维数组， get Ab波浪
        matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)
        graph_data += matrix_i

        degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float("inf")] = 0.  # [N]
        #返回以1D向量为对角线的二D数组
        degree_matrix = torch.diag(degree_matrix) #n,n

        return torch.mm(degree_matrix,graph_data) # D^(-1) * A = \hat(A)

def calculate_dgcn(adj):
    # adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return torch.Tensor(d_mat.dot(adj).astype(np.float32))

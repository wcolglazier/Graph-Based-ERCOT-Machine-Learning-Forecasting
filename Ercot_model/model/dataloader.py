import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


import os
import pandas as pd
import numpy as np
import scipy.sparse as sp

def load_adj(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)


    file_path = os.path.join(dataset_path, 'toy_data.csv')
    df = pd.read_csv(file_path, low_memory=False)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.set_index('timestamp')

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='all')

    returns = df.pct_change(fill_method=None).dropna()

    adj_dict = {}
    for target_node in returns.columns:
        final_corr_values = {
            col: returns[target_node].expanding().corr(returns[col]).iloc[-1] ** 4
            for col in returns.columns if col != target_node
        }
        adj_dict[target_node] = final_corr_values

    node_list = list(returns.columns)
    n_vertex = len(node_list)
    adj_matrix = np.zeros((n_vertex, n_vertex))

    for i, node_i in enumerate(node_list):
        for j, node_j in enumerate(node_list):
            if node_i != node_j:
                adj_matrix[i, j] = adj_dict[node_i].get(node_j, 0.0)

    adj = sp.csc_matrix(adj_matrix)



    return adj, n_vertex



def load_data(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'toy_data.csv'))

    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
    return train, val, test

def data_transform(data, n_his, n_pred, device):

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred

    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])

    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
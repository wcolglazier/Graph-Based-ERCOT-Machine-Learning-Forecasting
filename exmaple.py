import logging
import os
import argparse
import math
import warnings
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from model import utility, dataloader, Early_stop, models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# User set parameters
# Name of the folder the data is in
dataset = 'ercot'

# The number of historical time steps analyzed per set of predicted values
num_hist_ts = 20

# Number of future steps predicted simultaneously (e.g., 1: next step, 3: next three steps)
num_pred = 1

# number of epochs
epochs = 1000

# Number of epochs it will go without improvement before stopping
pat = 20

# Set random seed
seed = 40



def data_preparate(args, device):
    adj, n_vertex = dataloader.load_adj(dataset)
    gso = utility.calc_gso(adj, args.gso_type)

    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    dataset_path = 'data'
    dataset_path = os.path.join(dataset_path, dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'toy_data.csv')).shape[0]

    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    train, val, test = dataloader.load_data(dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = dataloader.data_transform(train, args.n_his, num_pred, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, num_pred, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, num_pred, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter


def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_his', type=int, default=num_hist_ts)
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=5)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu')
    parser.add_argument('--Ks', type=int, default=3)
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap')
    parser.add_argument('--enable_bias', type=bool, default=True)
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)

    args = parser.parse_args()

    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])

    return args, device, blocks


def prepare_model(args, blocks, n_vertex):
    loss = nn.MSELoss()
    es = Early_stop.EarlyStopping(delta=0.0,
                                  patience=pat,
                                  verbose=True,
                                  path=dataset + ".pt")


    model = models.GraphConv(args, blocks, n_vertex).to(device)


    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)


    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    return loss, es, model, optimizer, scheduler

def train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter):
    for epoch in range(epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = val(model, val_iter)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        es(val_loss, model)
        if es.early_stop:
            print("Early stopping")
            break

@torch.no_grad()
def val(model, val_iter):
    model.eval()

    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        loss = nn.MSELoss()
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)

@torch.no_grad() 
def test(zscore, loss, model, test_iter, args):
    model.load_state_dict(torch.load(dataset + ".pt"))
    model.eval()

    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)
    print(f'Test loss {test_MSE:.6f}  MAE {test_MAE:.6f}')

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args, device, blocks = get_parameters()
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
    train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter)
    test(zscore, loss, model, test_iter, args)


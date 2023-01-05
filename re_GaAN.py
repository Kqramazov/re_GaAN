import os
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
import copy
import pandas as pd
from tqdm import trange
import numpy as np
import torch.optim as optim
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import torch_scatter
import torch
import torch_geometric
torch_geometric.__version__


class GAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=2,
                 negative_slope=0.2, dropout=0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        # print("in:"+str(in_channels))
        self.out_channels = out_channels
        # print("out:"+str(out_channels))
        self.heads = heads
        # print("head:"+str(heads))
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = nn.Linear(
            self.in_channels, self.out_channels * self.heads)
        self.lin_r = self.lin_l
        self.att_r = Parameter(torch.Tensor(1, self.heads, self.out_channels))
        self.att_l = Parameter(torch.Tensor(1, self.heads, self.out_channels))

        self.final_lin = nn.Linear(
            self.in_channels+self.heads*self.out_channels, self.heads*self.out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size=None):

        H, C = self.heads, self.out_channels

        w_l = self.lin_l(x).view(-1, H, C)
        w_r = self.lin_r(x).view(-1, H, C)
        # print(w_r.shape)
        alpha_l = (w_l * self.att_l).sum(-1)
        alpha_r = (w_r * self.att_r).sum(-1)

        out = self.propagate(edge_index=edge_index, x=(w_l, w_r), alpha=(
            alpha_l, alpha_r), dim_size=size).view(-1, H * C)
        # print("in forward:")
        # print(out.shape)
        # print(x.shape)
        out1 = torch.cat((x.T, out.T)).T
        # print("in forward:")
        # print(out1.shape)
        out1 = self.final_lin(out1)
        return out1

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):

        # alpha = [E,H]
        # x = [N,H,C]
        H, C = self.heads, self.out_channels

        alpha = alpha_i + alpha_j
        # print("alpha:")
        # print(alpha.shape)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        alpha = alpha.unsqueeze(-1)  # alpha = [E,H,1]
        #alpha [E, H, C]
        # print("alpha:")
        # print(alpha.shape)
        # print("x_j")
        # print(x_j.shape)
        out = x_j * alpha  # [N,H,C] * [E,H,C](alpha进行广播)
        # print("in msg:")
        # print(out.shape)
        return out

    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, reduce='sum', dim=0)
        # print("in aggr")
        # print(out.shape)
        return out


class GaAN(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=2, m=8,
                 negative_slope=0.2, dropout=0., **kwargs):
        super(GaAN, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.m = m
        # print("in:"+str(in_channels))
        # print("out:"+str(out_channels))
        # print("head:"+str(heads))
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = nn.Linear(
            self.in_channels, self.out_channels * self.heads)
        self.lin_r = self.lin_l
        self.att_r = Parameter(torch.Tensor(1, self.heads, self.out_channels))
        self.att_l = Parameter(torch.Tensor(1, self.heads, self.out_channels))

        # GaAN layers
        self.g_lin = nn.Linear(
            2*self.in_channels+m, self.heads)
        self.m_lin = nn.Linear(self.in_channels, self.m)
        self.final_lin = nn.Linear(
            self.in_channels+self.heads*self.out_channels, self.heads*self.out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size=None):

        H, C = self.heads, self.out_channels

        w_l = self.lin_l(x).view(-1, H, C)
        w_r = self.lin_r(x).view(-1, H, C)
        # print(w_r.shape)
        alpha_l = (w_l * self.att_l).sum(-1)
        alpha_r = (w_r * self.att_r).sum(-1)

        m = self.m_lin(x)
        out, m, z = self.propagate(edge_index, Wx=(w_l, w_r), z=(x, x), alpha=(
            alpha_l, alpha_r), m=(m, m), dim_size=size)
        # 计算gate_i向量
        g_in = torch.cat((x.T, m.T, z.T)).T
        g_out = self.g_lin(g_in)
        g_i = torch.sigmoid(g_out)
        # print(g_i.shape)
        # print(out.shape)
        # 为各个head的结果一一乘上对应的g_i

        out = out * g_i.view(-1, H, 1)
        out = out.view(-1, H * C)
        # 拼接上了本层的数据x，假定是要这么做吧
        out_cat = torch.cat((x.T, out.T)).T
        out_cat = self.final_lin(out_cat)
        # print("in forward:")
        # print(out1.shape)
        return out_cat

    def message(self, Wx_j, z_j, alpha_j, alpha_i, m_j, index, ptr, size_i):

        H, C = self.heads, self.out_channels

        alpha = alpha_i + alpha_j
        # print("alpha:")
        # print(alpha.shape)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        alpha = alpha.unsqueeze(-1)  # alpha = [E,H,1]
        # alpha [E, H, C]
        # print("alpha:")
        # print(alpha.shape)
        # print("x_j")
        # print(x_j.shape)
        out = Wx_j * alpha  # [N,H,C] * [E,H,C](alpha进行广播)
        # print("in msg:")
        # print(out.shape)

        # GaAN修改部分的系数：由节点i和其邻居N_i计算得到自己的g_i

        return (out, m_j, z_j)

    def aggregate(self, inputs, index, dim_size=None):
        # print("in aggr")
        # print(inputs.shape)
        inputs, m, z = inputs
        out = torch_scatter.scatter(inputs, index, reduce='sum', dim=0)
        m = torch_scatter.scatter(m, index, reduce='max', dim=0)
        z = torch_scatter.scatter(z, index, reduce='mean', dim=0)
        # print(m.shape)
        # print(x.shape)
        # print(out.shape)
        return (out, m, z)


class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(args.heads * hidden_dim,
                      hidden_dim), nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim))

        self.dropout = args.dropout
        self.num_layers = args.num_layers

        self.emb = emb

    def build_conv_model(self, model_type):
        # if model_type == 'GraphSage':
        #     return GraphSage
        if model_type == 'GAT':
            return GAT
        elif model_type == 'GaAN':
            return GaAN

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        if self.emb == True:
            return x

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr,
                               weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr,
                              momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(
            filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


def train(dataset, args):

    print("Node task. test set size:", np.sum(dataset[0]['test_mask'].numpy()))
    print()
    test_loader = loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)

    # build model
    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes,
                     args)
    scheduler, opt = build_optimizer(args, model.parameters())
    # train
    losses = []
    test_accs = []
    best_acc = 0
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)
            # with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        losses.append(total_loss)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            test_accs.append(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(model)
        else:
            test_accs.append(test_accs[-1])

    return test_accs, losses, best_model, best_acc, test_loader


def test(loader, test_model, is_validation=False, save_model_preds=False, model_type=None):
    test_model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = test_model(data).max(dim=1)[1]
            label = data.y

        mask = data.val_mask if is_validation else data.test_mask
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = label[mask]

        if save_model_preds:
            print("Saving Model Predictions for Model Type", model_type)

            data = {}
            data['pred'] = pred.view(-1).cpu().detach().numpy()
            data['label'] = label.view(-1).cpu().detach().numpy()

            df = pd.DataFrame(data=data)
            # Save locally as csv
            df.to_csv('CORA-Node-' + model_type + '.csv', sep=',', index=False)

        correct += pred.eq(label).sum().item()

    total = 0
    for data in loader.dataset:
        total += torch.sum(data.val_mask if is_validation else data.test_mask).item()

    return correct / total


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


if 'IS_GRADESCOPE_ENV' not in os.environ:
    for args in [
        {'model_type': 'GAT', 'dataset': 'cora', 'num_layers': 2, 'heads': 1, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.5,
            'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01},
    ]:
        args = objectview(args)
        for model in ['GAT', 'GaAN']:
            args.model_type = model
            print("model:"+model)
            # Match the dimension.
            args.heads = 2

            if args.dataset == 'cora':
                dataset = Planetoid(root='/tmp/cora', name='Cora')
            else:
                raise NotImplementedError("Unknown dataset")
            test_accs, losses, best_model, best_acc, test_loader = train(
                dataset, args)

            print("Maximum test set accuracy: {0}".format(max(test_accs)))
            print("Minimum loss: {0}".format(min(losses)))

            # Run test for our best model to save the predictions!
            test(test_loader, best_model, is_validation=False,
                 save_model_preds=True, model_type=model)
            print()

            plt.title(dataset.name)
            plt.plot(losses, label="training loss" + " - " + args.model_type)
            plt.plot(test_accs, label="test accuracy" +
                     " - " + args.model_type)
        plt.legend()
        plt.show()

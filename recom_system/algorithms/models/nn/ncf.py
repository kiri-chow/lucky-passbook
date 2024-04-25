#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:54:58 2024

@author: anthony
"""
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


class NCFModel(nn.Module):
    """
    The Neural CF model

    """

    def __init__(self, in_features=20, out_features=1,
                 mlp_size=(30, 30, 10)):
        super().__init__()
        mlp_layers = [MLPLayer(in_features, mlp_size[0], short_cut=False)]
        for in_f, out_f in zip(mlp_size[:-1], mlp_size[1:]):
            mlp_layers.append(MLPLayer(in_f, out_f, short_cut=True))
        self.mlp = nn.Sequential(*mlp_layers)
        self.gmf = nn.Linear(in_features, mlp_size[-1])
        self.final = nn.Linear(mlp_size[-1] * 2, 1)
        self.last = nn.ReLU()

    def forward(self, x):
        y1 = self.mlp(x)
        y2 = self.gmf(x)
        y = torch.cat([y1, y2], dim=1)
        y = self.final(y)
        y = self.last(y)
        return y


class NCFDataset(Dataset):
    """
    Dataset for NCF Model

    """

    def __init__(self, data, svd, biased=True):
        self.data = data
        self.svd = svd
        self.biased = biased

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uid, iid, label = self.data[idx]
        uid = int(uid)
        iid = int(iid)
        user = self.svd.pu[uid]
        item = self.svd.qi[iid]

        feat = np.concatenate([user, item])

        if self.biased:
            mju = self.svd.trainset.global_mean
            label = label - self.svd.bi[iid] - self.svd.bu[uid] - mju

        return torch.Tensor(feat), torch.Tensor([label])


class NCFClf(NCFModel):
    """
    The classification variant of NCF Model

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last = nn.Sigmoid()


class NCFClfDataset(NCFDataset):
    def __init__(self, data, svd, biased=False, threshold=3):
        super().__init__(data, svd, biased=False)
        self.threshold = threshold

    def __getitem__(self, idx):
        feat, label = super().__getitem__(idx)
        label = (label > self.threshold).float()
        return feat, label


class MLPLayer(nn.Module):
    """
    A custom layer of MLP

    """

    def __init__(self, in_features, out_features, short_cut=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )
        self.short_cut = short_cut and in_features == out_features

    def forward(self, x):
        y = self.layer(x)
        if self.short_cut:
            y = x + y
        return y


def train_model(model, train_loader, test_loader, optimizer, loss_func,
                epochs=20, log_name='svd_nn'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter()
    losses = []
    for epoch in range(epochs):
        train_loss = train_once(
            model, tqdm(train_loader, f'{epoch+1}/{epochs}'),
            optimizer, loss_func, device)
        test_loss = test_once(model, test_loader, loss_func, device)
        writer.add_scalars(log_name, {'training': train_loss,
                                      'validation': test_loss}, epoch + 1)
        losses.append((train_loss, test_loss))
    return losses


def train_once(model, loader, optimizer, loss_func, device='cuda'):
    model.to(device)
    model.train()
    losses = []
    for feats, labels in loader:
        optimizer.zero_grad()
        preds = model(feats.to(device))
        loss = loss_func(preds, labels.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    loss = sum(losses) / len(losses)
    return loss


def test_once(model, loader, loss_func, device='cuda'):
    model.to(device)
    model.eval()
    losses = []
    with torch.no_grad():
        for feats, labels in loader:
            preds = model(feats.to(device))
            loss = loss_func(preds, labels.to(device))
            losses.append(loss.item())
    loss = sum(losses) / len(losses)
    return loss

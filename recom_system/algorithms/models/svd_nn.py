#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:59:34 2024

@author: anthony
"""
import os
import pickle
import random
from tqdm import tqdm
from surprise import SVD
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from recom_system.algorithms.models.base import BaseModel


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


class SvdNCF(BaseModel):
    """
    A combination of SVD and NCF Model

    """
    ncf_clazz = NCFModel

    data_clazz = NCFDataset

    def __init__(self, svd_params=None, ncf_params=None, fit_and_train=False,
                 biased=True):
        super().__init__()
        if not svd_params:
            svd_params = {}
        svd_params["biased"] = biased
        self.svd = SVD(**svd_params)
        if not ncf_params:
            ncf_params = {}
        self.ncf = self.ncf_clazz(**ncf_params)

        self.fit_and_train = fit_and_train

    def save(self, path):
        "save the model"
        svd = self.svd
        ncf = self.ncf
        path_svd = os.path.join(path, 'svd.pkl')
        path_ncf = os.path.join(path, 'ncf.pt')

        with open(path_svd, 'wb') as f:
            pickle.dump(svd, f)
        torch.save(ncf.state_dict(), path_ncf)
        return (path_svd, path_ncf)

    @classmethod
    def load(cls, path):
        path_svd = os.path.join(path, 'svd.pkl')
        path_ncf = os.path.join(path, 'ncf.pt')

        model = cls()
        with open(path_svd, 'rb') as f:
            model.svd = pickle.load(f)
        model.ncf.load_state_dict(torch.load(path_ncf))
        model.fit(model.svd.trainset, dry_run=True)
        return model

    def fit(self, trainset, dry_run=False):
        super().fit(trainset)
        if dry_run:
            return self

        self.svd.fit(trainset)

        if self.fit_and_train:
            self.train(
                torch.optim.Adam(self.ncf.parameters(), 1e-3),
                nn.MSELoss(), epochs=5,
            )

        return self

    def estimate(self, user_id, item_id):
        'estimate the rating'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        feature, bias = self.__build_feature(user_id, item_id)
        feature = torch.Tensor(feature).reshape(1, -1)

        with torch.no_grad():
            self.ncf.to(device)
            self.ncf.eval()
            pred = self.ncf(feature.to(device))
        pred = pred.item() + bias
        return pred

    def estimate_batch(self, user_id, item_idx):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features = []
        biases = []
        for iid in item_idx:
            feature, bias = self.__build_feature(user_id, iid)
            features.append(feature)
            biases.append(bias)
        features = torch.Tensor(features)

        with torch.no_grad():
            self.ncf.to(device)
            self.ncf.eval()
            preds = self.ncf(features.to(device))
            preds = preds.cpu().numpy()
        preds = preds.flatten()
        preds += np.array(biases)
        return preds

    def train(self, optimizer, loss_func, epochs=20, batch_size=64):
        "train the NCF Model"
        trainset, testset = self._build_dataset()
        trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
        testloader = DataLoader(testset, shuffle=False, batch_size=batch_size)

        return train_model(self.ncf, trainloader, testloader, optimizer,
                           loss_func=loss_func, epochs=epochs)

    def _build_dataset(self, train_size=0.8):
        index = list(range(self.trainset.n_ratings))
        random.shuffle(index)
        cut = int(len(index) * train_size)

        data = np.array(list(self.trainset.all_ratings()))
        traindata = data[index[: cut]]
        testdata = data[index[cut:]]

        trainset = self.data_clazz(traindata, self.svd)
        testset = self.data_clazz(testdata, self.svd)
        return trainset, testset

    def __build_feature(self, uid, iid):
        try:
            user = self.svd.pu[uid]
            bu = self.svd.bu[uid]
        except IndexError:
            user = np.zeros(self.svd.n_factors)
            bu = 0

        try:
            item = self.svd.qi[iid]
            bi = self.svd.bi[iid]
        except IndexError:
            item = np.zeros(self.svd.n_factors)
            bi = 0

        feat = np.concatenate([user, item])
        if self.svd.biased:
            bias = bu + bi + self.svd.trainset.global_mean
        else:
            bias = 0
        return feat, bias


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


class SvdNCFClf(SvdNCF):
    "The classification variant of SVD NCF Model"

    ncf_clazz = NCFClf

    data_clazz = NCFClfDataset

    def estimate(self, user_id, item_id):
        pred = super().estimate(user_id, item_id)

        lower, upper = self.trainset.rating_scale
        return pred * (upper - lower) + lower

    def estimate_batch(self, user_id, item_idx):
        preds = super().estimate_batch(user_id, item_idx)

        lower, upper = self.trainset.rating_scale
        return preds * (upper - lower) + lower


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

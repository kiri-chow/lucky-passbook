#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:59:34 2024

@author: anthony
"""
import os
import pickle
import random
from surprise import SVD
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from recom_system.algorithms.models.base import BaseModel
from recom_system.algorithms.models.nn.ncf import NcfModel, train_model


class SvdNCF(BaseModel):
    """
    A combination of SVD and NCF Model

    """

    def __init__(self, svd_params=None, ncf_params=None, fit_and_train=False,
                 biased=True, verbose=True, n_epochs=10):
        super().__init__()
        if not svd_params:
            svd_params = {"n_factors": 10, "n_epochs": 20}
        svd_params["biased"] = biased
        self.svd = SVD(**svd_params)
        if not ncf_params:
            ncf_params = {}
        ncf_params['in_features'] = self.svd.n_factors * 2
        self.ncf = NcfModel(**ncf_params)

        self.verbose = verbose
        self.n_epochs = n_epochs
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
                nn.MSELoss(), epochs=self.n_epochs,
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

        trainset = SvdNcfDataset(traindata, self.svd)
        testset = SvdNcfDataset(testdata, self.svd)
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


class SvdNcfDataset(Dataset):
    """
    Dataset for SVD-NCF Model

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

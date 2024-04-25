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
from torch.utils.data import DataLoader
from recom_system.algorithms.models.base import BaseModel
from recom_system.algorithms.models.nn.ncf import (
    NCFModel, NCFClf, NCFDataset, NCFClfDataset, train_model as train_ncf)


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

        return train_ncf(self.ncf, trainloader, testloader, optimizer,
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

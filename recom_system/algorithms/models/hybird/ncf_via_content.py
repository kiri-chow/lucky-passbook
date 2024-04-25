#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:05:37 2024

@author: anthony
"""
import os
import random
import pickle
from math import isnan
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from recom_system.algorithms.models.base import BaseModel
from recom_system.algorithms.models.vector_based.sentence_embeddings import build_item_matrix
from recom_system.algorithms.models.nn.ncf import NcfModel, train_model


class NcfViaContent(BaseModel):
    """
    A combination of NCF and content Model

    """
    name = 'ncf_via_content'

    def __init__(self, item_profiles, pca=100, threshold=3, ncf_params=None,
                 fit_and_train=False, verbose=True, n_epochs=10):
        self.pca = None
        if pca:
            self.pca = PCA(pca)
        if self.pca:
            self.item_matrix = pd.DataFrame(self.pca.fit_transform(item_profiles),
                                            index=item_profiles.index)
        else:
            self.item_matrix = item_profiles
        self.threshold = threshold

        if not ncf_params:
            ncf_params = {}
        self.n_features = len(self.item_matrix.iloc[0])
        ncf_params['in_features'] = self.n_features * 2
        self.ncf = NcfModel(**ncf_params)
        self.fit_and_train = fit_and_train
        self.verbose = verbose
        self.n_epochs = n_epochs

    def save(self, path):
        "save the model"
        pca = self.pca
        ncf = self.ncf

        path_pca = os.path.join(path, f'{self.name}_pca.pkl')
        path_ncf = os.path.join(path, f'{self.name}_ncf.pt')

        with open(path_pca, 'wb') as f:
            pickle.dump(pca, f)
        torch.save(ncf.state_dict(), path_ncf)
        return path

    def load(self, path):
        path_ncf = os.path.join(path, f'{self.name}_ncf.pt')
        return self.ncf.load_state_dict(torch.load(path_ncf))

    def fit(self, trainset):
        super().fit(trainset)

        self.users = np.asarray([
            self._build_user_profile(uid)
            for uid in range(self.trainset.n_users)
        ])

        self.bu, self.bi = self.__compute_bias()

        if self.fit_and_train:
            self.train(
                torch.optim.Adam(self.ncf.parameters(), 1e-3),
                nn.MSELoss(), epochs=self.n_epochs,
            )

        return self

    def _build_user_profile(self, user_id):
        user_ratings = np.asarray(self.trainset.ur[user_id])

        # only consider the positive items
        user_ratings = user_ratings[user_ratings[:, 1] > self.threshold]
        index, ratings = user_ratings.T

        # build user profile
        item_profiles = self.get_item_profile(index)
        if len(item_profiles) == 0:
            return np.zeros(self.n_features)
        user_profile = ratings.dot(item_profiles)

        # normalization
        norm = np.linalg.norm(user_profile, 1)
        if norm > 0:
            user_profile /= norm
        return user_profile

    def get_item_profile(self, iidx):
        riidx = [x for x in self.to_raw_iid(iidx) if not isnan(x)]
        if len(riidx) == 0:
            return []
        items = self.item_matrix.loc[riidx].values
        return items

    def __compute_bias(self):
        users, items, ratings = np.asarray(list(self.trainset.all_ratings())).T
        users = users.astype(int)
        items = items.astype(int)

        bu = np.zeros(self.trainset.n_users)
        for uid in np.unique(users):
            bu[uid] = ratings[uid == users].mean() - self.trainset.global_mean

        bi = np.zeros(self.trainset.n_items)
        for iid in np.unique(items):
            bi[iid] = ratings[iid == items].mean() - self.trainset.global_mean
        return bu, bi

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
                           loss_func=loss_func, epochs=epochs, log_name='ncf_content')

    def _build_dataset(self, train_size=0.8):
        index = list(range(self.trainset.n_ratings))
        random.shuffle(index)
        cut = int(len(index) * train_size)

        data = np.array(list(self.trainset.all_ratings()))
        traindata = data[index[: cut]]
        testdata = data[index[cut:]]

        trainset = NcfContentDataset(traindata, self)
        testset = NcfContentDataset(testdata, self)
        return trainset, testset

    def __build_feature(self, uid, iid):
        try:
            user = self.users[uid]
            bu = self.bu[uid]
        except IndexError:
            user = np.zeros(self.n_features)
            bu = 0

        try:
            item = self.item_matrix.loc[self.to_raw_iid(iid)].values
        except (IndexError, ValueError):
            item = np.zeros(self.n_features)
        try:
            bi = self.bi[iid]
        except IndexError:
            bi = 0

        feat = np.concatenate([user, item])
        bias = bu + bi + self.trainset.global_mean
        return feat, bias


class NcfContentDataset(Dataset):
    """
    Dataset for NCF via Content Model

    """

    def __init__(self, data, algo, biased=True):
        self.data = data
        self.algo = algo
        self.biased = biased

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uid, iid, label = self.data[idx]
        uid = int(uid)
        iid = int(iid)
        user = self.algo.users[uid]
        item = self.algo.get_item_profile([iid])[0]

        feat = np.concatenate([user, item])

        if self.biased:
            mju = self.algo.trainset.global_mean
            label = label - self.algo.bi[iid] - self.algo.bu[uid] - mju

        return torch.Tensor(feat), torch.Tensor([label])

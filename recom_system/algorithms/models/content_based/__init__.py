#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:09:41 2024

@author: anthony
"""
from surprise import AlgoBase
import numpy as np


class ContentBasedModel(AlgoBase):
    """
    content baed model

    params
    ------
    item_matrix : pd.DataFrame,
        the item - features dataframe.
    threshold : int,
        the threshold of a positve rating.
    fit_and_build : bool,
        build all the users' profiles in fitting.

    """

    def __init__(self, item_matrix, threshold=3, fit_and_build=False):
        super().__init__()
        self.item_matrix = item_matrix
        self.threshold = threshold
        self.fit_and_build = fit_and_build
        self.user_profiles = {}

    def fit(self, trainset):
        super().fit(trainset)
        self.to_raw_iid = np.frompyfunc(trainset.to_raw_iid, 1, 1)
        self.to_inner_iid = np.frompyfunc(trainset.to_inner_iid, 1, 1)

        if self.fit_and_build:
            self.user_profiles.update(dict(map(
                lambda x: (x, self._get_user_profile(x)),
                range(self.trainset.n_users),
            )))

        return self

    def estimate(self, user_id, item_id):
        "predict the rating of given user and item"
        if self.fit_and_build:
            user_profile = self.user_profiles[user_id]
        else:
            user_profile = self._get_user_profile(user_id)
        item_profile = self.item_matrix.loc[self.to_raw_iid(item_id)]
        return user_profile.dot(item_profile)

    def estimate_batch(self, user_id, item_idx):
        "estimate a batch of items for a single user"
        if self.fit_and_build:
            user_profile = self.user_profiles[user_id]
        else:
            user_profile = self._get_user_profile(user_id)
        item_profile = self.item_matrix.loc[self.to_raw_iid(item_idx)]
        return user_profile.dot(item_profile.T)

    def _get_user_profile(self, user_id):
        "build user_profile based on ratings"
        user_ratings = np.array(self.trainset.ur[user_id])

        # only consider the positive items
        user_ratings = user_ratings[user_ratings[:, 1] > self.threshold]
        index, ratings = user_ratings.T

        # build user profile
        index = self.to_raw_iid(index)
        items = self.item_matrix.loc[index].values
        user_profile = ratings.dot(items)

        # normalization
        norm = np.linalg.norm(user_profile)
        if norm == 0:
            return np.zeros_like(self.item_matrix.iloc[0])
        user_profile /= norm
        return user_profile

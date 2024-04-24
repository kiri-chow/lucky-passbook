#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:48:49 2024

@author: anthony
"""
from math import isnan
import numpy as np
from sklearn.neighbors import KDTree
from recom_system.algorithms.models.base import BaseModel, PredictionImpossible


class KNNViaContent(BaseModel):
    """
    use content-based features for user-based KNN

    params
    ------
    item_matrix : pd.DataFrame,
        the item - features dataframe.
    k : int,
        the number of neighbors
    min_k : int,
        minimal number of neighbors has rated for the item.
    threshold : float,
        the threshold of a positve rating.
    norm : int,
        order of user profile's norm

    """

    def __init__(self, item_matrix, k=50, min_k=3, threshold=3, norm=2):
        super().__init__()
        self.item_matrix = item_matrix
        self.k = k
        self.min_k = min_k
        self.threshold = threshold
        self.norm = norm

    def fit(self, trainset):
        super().fit(trainset)

        # build user profiles
        self.user_profiles_ = np.asarray([
            self._build_user_profile(uid)
            for uid in range(self.trainset.n_users)
        ])

        # build kd tree
        self.tree_ = KDTree(self.user_profiles_, self.k)

        return self

    def estimate(self, user_id, item_id):
        # get k neighbors
        user_profile, neighbors = self.__get_neighbors(user_id)

        # filter neighbors by item
        item_ratings = dict(self.trainset.ir[item_id])
        actual_neighbors = list(set(item_ratings) & set(neighbors))
        if len(actual_neighbors) < self.min_k:
            raise PredictionImpossible("No enough neighbors rated this item")

        # compute rating for user
        cosines = user_profile.dot(self.user_profiles_[actual_neighbors].T)
        delta_ratings = np.array([
            item_ratings[uid] - self.__get_user_mean(uid)
            for uid in actual_neighbors])
        sum_sim = np.abs(cosines).sum() + 1e-10
        user_mean = self.__get_user_mean(user_id)

        pred = user_mean + cosines.dot(delta_ratings) / sum_sim
        return pred

    def __get_user_mean(self, user_id):
        return np.mean([r for i, r in self.trainset.ur[user_id]])

    def __get_neighbors(self, user_id):
        try:
            user_profile = self.user_profiles_[user_id]
        except ValueError:
            raise PredictionImpossible("Unknown user")

        # get neighbors
        _, neighbors = self.tree_.query(
            user_profile.reshape(1, -1), self.k + 1)
        neighbors = neighbors[0]
        return user_profile, neighbors

    def _build_user_profile(self, user_id):
        user_ratings = np.asarray(self.trainset.ur[user_id])

        # only consider the positive items
        user_ratings = user_ratings[user_ratings[:, 1] > self.threshold]
        index, ratings = user_ratings.T

        # build user profile
        item_profiles = self._get_item_profile(index)
        if len(item_profiles) == 0:
            return np.zeros_like(self.item_matrix.iloc[0])
        user_profile = ratings.dot(item_profiles)

        # normalization
        norm = np.linalg.norm(user_profile, self.norm)
        if norm > 0:
            user_profile /= norm
        return user_profile

    def _get_item_profile(self, iidx):
        riidx = [x for x in self.to_raw_iid(iidx) if not isnan(x)]
        if len(riidx) == 0:
            return []
        items = self.item_matrix.loc[riidx].values
        return items

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:09:41 2024

@author: anthony
"""
from math import isnan
import numpy as np
from recom_system.algorithms.models.base import BaseModel, PredictionImpossible


class VectorBasedModel(BaseModel):
    """
    Content based model for vector-like data (dense)

    params
    ------
    item_profiles : pd.DataFrame,
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

        # build user profile
        self.user_profiles.clear()
        if self.fit_and_build:
            for user_id in range(self.trainset.n_users):
                try:
                    self.user_profiles[user_id] = self._get_user_profile(
                        user_id)
                except PredictionImpossible:
                    continue

        return self

    def _get_user_profile(self, user_id):
        "build user_profile based on ratings"
        try:
            return self.user_profiles[user_id]
        except:
            pass

        try:
            user_ratings = np.array(self.trainset.ur[user_id])
        except KeyError:
            raise PredictionImpossible("Unknown user")

        # only consider the positive items
        user_ratings = user_ratings[user_ratings[:, 1] > self.threshold]
        index, ratings = user_ratings.T

        # build user profile
        item_profiles = self._get_item_profile(index)
        user_profile = ratings.dot(item_profiles)

        # normalization
        l1_norm = sum(user_profile)
        if l1_norm > 0:
            user_profile /= l1_norm
        self.user_profiles[user_id] = user_profile
        return user_profile

    def _get_item_profile(self, iidx):
        riidx = [x for x in self.to_raw_iid(iidx) if not isnan(x)]
        if len(riidx) == 0:
            raise PredictionImpossible("Unknown item")
        items = self.item_matrix.loc[riidx].values
        return items

    def estimate(self, user_id, item_id):
        "predict the rating of given user and item"
        try:
            user_profile = self._get_user_profile(user_id)
            item_profile = self._get_item_profile([item_id])[0]
            return user_profile.dot(item_profile)
        except:
            return self.default_prediction()

    def estimate_batch(self, user_id, item_idx):
        "estimate a batch of items for a single user"
        try:
            user_profile = self._get_user_profile(user_id)
            item_profiles = self._get_item_profile(item_idx)
            return user_profile.dot(item_profiles.T)
        except:
            return [self.default_prediction() for _ in item_idx]

    def default_prediction(self):
        return 0.5

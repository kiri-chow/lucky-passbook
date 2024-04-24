#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 01:05:07 2024

@author: anthony
"""
from math import isnan
from recom_system.algorithms.models.base import BaseModel, PredictionImpossible


class DictBasedModel(BaseModel):
    """
    Content based model for dict-like data (sparse)

    params
    ------
    item_profiles : dict,
        the item - features dict
    threshold : int,
        the threshold of a positve rating.
    fit_and_build : bool,
        build all the users' profiles in fitting.

    """

    def __init__(self, item_profiles, threshold, fit_and_build=False):
        super().__init__()
        self.item_profiles = item_profiles
        self.threshold = threshold
        self.fit_and_build = fit_and_build
        self.user_profiles = {}

    def fit(self, trainset):
        super().fit(trainset)

        # build user profile
        self.user_profiles.clean()
        if self.fit_and_build:
            for user_id in range(self.trainset.n_users):
                try:
                    self.user_profiles[user_id] = self._get_user_profile(
                        user_id)
                except PredictionImpossible:
                    continue

        return self

    def _get_user_profile(self, user_id):
        try:
            return self.user_profiles[user_id]
        except:
            pass

        # build user profile
        user_profile = {}
        try:
            user_ratings = self.trainset.ur[user_id]
        except KeyError:
            raise PredictionImpossible("Unknown user")
        for item_id, rating in user_ratings:
            if rating <= self.threshold:
                continue
            try:
                item_profile = self._get_item_profile(item_id)
            except PredictionImpossible:
                continue
            for key, value in item_profile:
                user_profile[key] = (user_profile.get(key, 0) +
                                     value * rating)
        self.user_profiles[user_id] = user_profile
        return user_profile

    def _get_item_profile(self, riid):
        iid = self.to_raw_iid(riid)
        if isnan(iid):
            raise PredictionImpossible("Unknown item")
        return self.item_profiles[iid]

    def estimate(self, user_id, item_id):
        user_profile = self._get_user_profile(user_id)
        item_profile = self._get_item_profile(item_id)

        score = 0
        for key, weight in user_profile:
            score += item_profile.get(key, 0) * weight
        return score

    def default_prediction(self):
        "return a neutral cosine similarity"
        return 0

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:23:25 2024

@author: anthony
"""
from collections.abc import Iterable
from math import nan, isnan
import numpy as np
from surprise import AlgoBase, PredictionImpossible


class BaseModel(AlgoBase):
    """
    base model has necessary methods

    """

    def fit(self, trainset):
        super().fit(trainset)
        self.to_raw_uid = np.frompyfunc(trainset.to_raw_uid, 1, 1)
        self.to_raw_iid = np.frompyfunc(trainset.to_raw_iid, 1, 1)

    def estimate_batch(self, user_id, item_idx):
        if isnan(user_id):
            return [self.default_prediction() for _ in item_idx]

        return np.asarray([
            (self.default_prediction() if isnan(iid)
             else self.estimate(user_id, iid))
            for iid in item_idx])

    def __base_conv_id(self, func, x):
        if isinstance(x, Iterable):
            return [self.__base_conv_id(func, xx) for xx in x]
        try:
            return func(int(x))
        except (ValueError, KeyError):
            return nan

    def to_raw_iid(self, x):
        return self.__base_conv_id(self.trainset.to_raw_iid, x)

    def to_raw_uid(self, x):
        return self.__base_conv_id(self.trainset.to_raw_uid, x)

    def to_inner_iid(self, x):
        return self.__base_conv_id(self.trainset.to_inner_iid, x)

    def to_inner_uid(self, x):
        return self.__base_conv_id(self.trainset.to_inner_uid, x)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 11:40:09 2024

@author: anthony
"""
from math import isnan
from collections.abc import Iterable
import nevergrad as ng
import numpy as np
from .base import BaseModel
from surprise import PredictionImpossible


class WeightedModel(BaseModel):
    """
    A weighted hybrid model

    params
    ------
    models : list of model,
        base models
    weights : list of float or None,
        weights of base models, None means uniformed weights
    auto_weight : int,
        the evolution times for weights optimization,
        0 means to use the given weights
    fixed : bool,
        use fixed weight for all of the users
    verbose : bool,
        print dynamic weights or not

    """

    def __init__(self, models, weights=None, auto_weight=1000,
                 fixed=False, verbose=False,):
        super().__init__()
        self.models = models
        if weights is None:
            weights = [1 / len(self.models) for _ in self.models]
        self.weights = np.asarray(weights)
        self.auto_weight = auto_weight
        self.verbose = verbose
        self.fixed = fixed
        self.user_weights = {}

    def fit(self, trainset):
        super().fit(trainset)

        for model in self.models:
            model.fit(trainset)

        if self.auto_weight and self.fixed:
            self._determine_fixed()

        return self

    def default_prediction(self):
        return self.weights.dot([model.default_prediction()
                                 for model in self.models])

    def _determine_fixed(self):
        user_ratings = np.asarray(list(self.trainset.all_ratings()))
        target = user_ratings[:, 2]
        preds = np.array([
            [self._handle_model_est(model, uid, iid)
             for uid, iid in user_ratings[:, :2].astype(int)]
            for model in self.models
        ]).T

        result = self.__auto_weight_by_pso(target, preds)
        if self.verbose:
            print(
                f"The weights for whole trainset is {result.value}, loss = {result.loss:.2f}")
        self.weights = np.asarray(result.value)

    def _determine_weights(self, user_id):
        if not self.auto_weight or self.fixed:
            return self.weights

        user_ratings = self.trainset.ur[user_id]
        if not user_ratings:
            return self.weights

        target = np.array([r for _, r in user_ratings])
        preds = np.array([
            [self._handle_model_est(model, user_id, iid)
             for iid, _ in user_ratings]
            for model in self.models
        ]).T

        result = self.__auto_weight_by_pso(target, preds)
        if self.verbose:
            print(
                f"The weights for <user {user_id}> is {result.value}, loss = {result.loss:.2f}")

        return result.value

    def __auto_weight_by_pso(self, target, preds):
        # use PSO for the best weight
        def objective_func(x):
            # minimize the RMSE
            return np.sqrt(((target - (preds * x).sum(1))**2).sum())

        optim = ng.optimizers.RealSpacePSO(
            parametrization=ng.p.Array(shape=(len(self.models), ),
                                       lower=0, upper=1),
            budget=self.auto_weight,
        )
        result = optim.minimize(objective_func)
        return result

    def estimate(self, user_id, item_id):
        "predict the rating of given user and item"
        if isnan(user_id) or isnan(item_id):
            return self.default_prediction()
        preds = [self._handle_model_est(model, user_id, item_id)
                 for model in self.models]

        # specify the weights for each user
        weights = self._determine_weights(user_id)
        return (weights * preds).sum()

    def estimate_batch(self, user_id, item_idx):
        "estimate a batch of items for a single user"
        if isnan(user_id):
            return self.default_prediction()

        assert isinstance(item_idx, Iterable)
        weights = self._determine_weights(user_id)
        preds = np.array([[self._handle_model_est(model, user_id, iid)
                           for iid in item_idx]
                          for model in self.models]).T
        return (weights * preds).sum(1)

    def _handle_model_est(self, model, uid, iid):
        try:
            est = model.estimate(uid, iid)
            if isinstance(est, Iterable):
                return est[0]
            return est
        except PredictionImpossible:
            return self.default_prediction()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 11:40:09 2024

@author: anthony
"""
from collections.abc import Iterable
import nevergrad as ng
import numpy as np
from surprise import AlgoBase


class WeightedModel(AlgoBase):
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
    fit_and_build : bool,
        build all the users' profiles in fitting.
    verbose : bool,
        print dynamic weights or not

    """

    def __init__(self, models, weights=None, auto_weight=1000,
                 fit_and_build=False, verbose=False):
        AlgoBase.__init__(self)
        self.models = models
        if weights is None:
            weights = [1 / len(self.models) for _ in self.models]
        self.weights = np.asarray(weights)
        self.auto_weight = auto_weight
        self.verbose = verbose
        self.fit_and_build = fit_and_build
        self.user_weights = {}

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        for model in self.models:
            model.fit(trainset)

        if self.auto_weight and self.fit_and_build:
            self.user_weights.update(dict(map(
                lambda x: (x, self._determine_weights(x)),
                range(self.trainset.n_users),
            )))

        return self

    def default_prediction(self):
        return 0

    def _determine_weights(self, user_id):
        user_ratings = self.trainset.ur[user_id]
        if not user_ratings:
            return self.weights

        target = np.array([r for _, r in user_ratings])
        preds = np.array([
            [_handle_model_est(model.estimate(user_id, iid))
             for iid, _ in user_ratings]
            for model in self.models
        ]).T

        # use PSO for the best weight
        def objective_func(x):
            # minimize the RMSE
            return np.sqrt(((target - (preds * x).sum(1))**2).sum())

        optim = ng.optimizers.RealSpacePSO(
            parametrization=ng.p.Array(shape=(len(self.models), ),
                                       lower=0, upper=1),
            budget=1000,
        )
        result = optim.minimize(objective_func)

        if self.verbose:
            print(
                f"The weights for <user {user_id}> is {result.value}, loss = {result.loss:.2f}")

        return result.value

    def estimate(self, user_id, item_id):
        "predict the rating of given user and item"
        preds = [_handle_model_est(model.estimate(user_id, item_id))
                 for model in self.models]

        # specify the weights for each user
        if self.auto_weight:
            if self.fit_and_build:
                weights = self.user_weights[user_id]
            else:
                weights = self._determine_weights(user_id)
        else:
            weights = self.weights

        return (weights * preds).sum()

    def estimate_batch(self, user_id, item_idx):
        "estimate a batch of items for a single user"
        assert isinstance(item_idx, Iterable)

        if self.auto_weight:
            if self.fit_and_build:
                weights = self.user_weights[user_id]
            else:
                weights = self._determine_weights(user_id)
        else:
            weights = self.weights

        preds = np.array([[_handle_model_est(model.estimate(user_id, iid))
                           for iid in item_idx]
                          for model in self.models]).T
        return (weights * preds).sum(1)


def _handle_model_est(est):
    if isinstance(est, Iterable):
        return est[0]
    return est

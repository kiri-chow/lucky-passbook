#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 11:40:09 2024

@author: anthony
"""
from surprise import AlgoBase, KNNWithMeans, SVDpp



class WeightedModel(AlgoBase):
    """
    A weighted hybrid model

    params
    ------
    models : list of model,
        base models
    weights : list of float or None,
        weights of base models, None means uniform weights
    auto_weight : bool,
        automatically determine the weight?

    """

    def __init__(self, models, weights=None, auto_weight=False):
        AlgoBase.__init__(self)
        self.models = models
        if weights is None:
            self.weights = [1 / len(self.models) for _ in self.models]
        else:
            weights = weights
        self.auto_weight = auto_weight

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        for model in self.models:
            model.fit(trainset)

        return self

    def _determine_weights(self, trainset):
        pass

    def estimate(self, user_id, item_id):

        scores_sum = 0
        weights_sum = 0

        for i in range(len(self.models)):
            # 3*1/4+4*3/4 laga ra
            scores_sum += self.models[i].estimate(
                user_id, item_id) * self.weights[i]
            weights_sum += self.weights[i]  # always becomes one

        return scores_sum / weights_sum

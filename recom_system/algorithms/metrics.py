#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 00:37:25 2024

@author: anthony
"""
from math import isnan
import numpy as np
from surprise.model_selection import KFold


def cross_validate(algo, data, measures=['rmse'], cv=5):
    kf = KFold(cv)
    results = {k: [] for k in measures}
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        result = evaluate(algo, testset, measures)
        for key, value in result.items():
            results[key].append(value)
    return results


def evaluate(algo, testset, measures=['rmse']):
    """
    return the test results for the algorithm

    """
    results = {k: [] for k in measures}
    if hasattr(algo, 'estimate_batch'):
        testset = np.asarray(testset)
        users = testset[:, 0]
        for user in np.unique(users):
            index = user == users
            items, y_true = testset[index].T[1:3]
            uid = algo.to_inner_uid(user)
            iidx = algo.to_inner_iid(items)
            y_pred = algo.estimate_batch(uid, iidx)
            for name in measures:
                results[name].append(
                    NAME_TO_FUNC.get(name, name)(y_true, y_pred))
    else:
        for row in testset:
            user, item, y_true = row[:3]
            uid = algo.to_inner_uid(user)
            iid = algo.to_inner_iid(item)
            if isnan(uid) or isnan(iid):
                y_pred = algo.default_prediction()
            else:
                y_pred = algo.estimate(uid, iid)
            for name in measures:
                results[name].append(
                    NAME_TO_FUNC.get(name, name)(y_true, y_pred))

    # compute the mean for all records
    mean_results = {}
    for name, values in results.items():
        if name == 'rmse':
            value = (sum(values) / len(testset))**0.5
        else:
            value = np.mean(values)
        mean_results[name] = value
    return mean_results


def se(y_true, y_pred):
    return ((y_true - y_pred)**2).sum()


def rmse(y_true, y_pred):
    '''
    returns
    -------
    rmse : float

    '''
    return ((y_true - y_pred)**2).mean()**0.5


def ndcg(y_true, y_pred, threshold=0):
    """
    params
    ------
    threshold : int,
        0 : compute nDCG with true ratings;
        other value : the threshold to determine a item hit the user or not.

    ndcg : float

    """
    ord_true = np.argsort(-y_true)
    ord_pred = np.argsort(-y_pred)
    decays = np.log2(np.arange(len(y_true)) + 2)

    if threshold:
        dcg = (y_true[ord_pred] > threshold / decays).sum()
        idcg = (y_true[ord_true] > threshold / decays).sum()
    else:
        dcg = (y_true[ord_pred] / decays).sum()
        idcg = (y_true[ord_true] / decays).sum()
    return dcg / (idcg + 1e-10)


def precision_recall(y_true, y_pred, n=10, threshold=3):
    '''
    returns
    -------
    precision@n, recall@n : float

    '''
    index = np.argsort(-y_pred)[:n]
    n_hits = (y_true[index] > threshold).sum()
    precision = n_hits / min(n, len(index) + 1e-10)
    recall = n_hits / ((y_true > threshold).sum() + 1e-10)
    return precision, recall


NAME_TO_FUNC = {
    'rmse': se,
    'ndcg': ndcg,
    'precision': lambda *x: precision_recall(*x)[0],
    'recall': lambda *x: precision_recall(*x)[1],
}

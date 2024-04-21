#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 11:49:34 2024

@author: anthony
"""
from surprise import SVDpp, KNNWithMeans
from recom_system.algorithms.preprocessing import get_ratings_datasets
from recom_system.algorithms.models.weighted_model import WeightedModel


knn = KNNWithMeans(
    k=50, min_k=3, verbose=False,
    sim_options={'user_based': False, 'name': 'cosine'},
)

svd = SVDpp(n_factors=10, n_epochs=20)


if __name__ == '__main__':
    trainset, testset = get_ratings_datasets()
    print(f'{len(trainset.raw_ratings)=}, {len(testset)=}')

    fulltrainset = trainset.build_full_trainset()
    
    model = WeightedModel([knn, svd])
    model.fit(fulltrainset)

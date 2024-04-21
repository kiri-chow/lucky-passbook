#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 11:49:34 2024

@author: anthony
"""
from surprise import SVDpp, KNNWithMeans
from recom_system.algorithms.preprocessing import get_ratings_datasets


if __name__ == '__main__':
    trainset, testset = get_ratings_datasets()
    print(f'{len(trainset.raw_ratings)=}, {len(testset)=}')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:48:46 2024

@author: anthony
"""

import random
from math import ceil
from surprise.dataset import Dataset, Reader
from recom_system.algorithms.io import get_ratings


READER = Reader(rating_scale=(1, 5))


def get_ratings_datasets(df=None, train_size=0.5, min_record=5,
                         drop_zero=True, random_state=42):
    """
    return the train set and test set for training algorithm

    params
    ------
    df : pd.DataFrame or None
        the DataFrame for preprocessing, None means use all record in database.
    train_size : float (0, 1)
        the percentage of train set's size
    min_record : int,
        minimal records of data
    drop_zero : bool, 
        drop the 0 rating or not.
    random_state : int,
        the random state for result consistency

    returns
    -------
    train_set : surprise.Trainset,
        the training set
    test_set : list,
        the test set

    """
    if df is None:
        df = get_ratings()

    if drop_zero:
        df = df[df.rating > 0]

    df = _filter_ratings(df, min_record)

    df_train, df_test = split_training_and_test(df, train_size, random_state)

    train_set = Dataset.load_from_df(
        df_train[['user_id', 'book_id', 'rating']], READER
    )

    test_set = df_test[['user_id', 'book_id', 'rating']].values
    return train_set, test_set


def _filter_ratings(df, min_record=5):
    "filter users and items satifying the min record"
    count_users = df.groupby(df.user_id).user_id.count()
    count_items = df.groupby(df.book_id).book_id.count()

    user_indices = count_users[count_users >= min_record].index
    item_indices = count_items[count_items >= min_record].index

    df = df[df.user_id.isin(user_indices) & df.book_id.isin(item_indices)]
    return df


def split_training_and_test(df, train_size=0.5, random_state=42):
    "split the df into training and test df"
    # set the state for consistent result
    random.seed(random_state)

    train_indices = set()

    # ensure train set includes half records for every user
    for user_id, indices in df.groupby(df.user_id).groups.items():
        size = ceil(len(indices) * train_size)
        train_indices |= set(random.sample(list(indices), size))

    # ensure train set includes half records for every item
    for item_id, indices in df.groupby(df.book_id).groups.items():
        indices = set(indices)
        included = indices & train_indices
        size_remaining = len(included) - ceil(len(indices) * train_size)

        if size_remaining > 0:
            pool = list(indices - train_indices)
            if len(pool) < size_remaining:
                train_indices |= set(pool)
            else:
                train_indices |= set(random.sample(pool, size_remaining))

    # split into 2 sets
    train_indices = list(train_indices)
    df_train = df.loc[train_indices]
    df_test = df.drop(index=train_indices)
    return df_train, df_test

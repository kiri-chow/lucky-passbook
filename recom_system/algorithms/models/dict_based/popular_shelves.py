#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:25:04 2024

@author: anthony
"""
import numpy as np
import pandas as pd


def build_item_matrix(books, tf_n=5, final_n=200):
    data = books.popular_shelves.apply(_get_popular_shelves, args=(tf_n, ))
    df = pd.DataFrame(data.to_list())

    # compute IDF
    idf = np.log(len(df) / (~df.isna()).sum(0))

    # compute TF-IDF
    df = (df.T / df.max(1)).T * idf

    if final_n:
        counts = df.sum(0).sort_values(ascending=False).iloc[:final_n]
        df = df[counts.index]

    df.fillna(0, inplace=True)
    df.index = books.id
    return df


def _get_popular_shelves(field, top_n):
    data = sorted(((x['name'], int(x['count']))
                  for x in field), key=lambda x: -x[1])
    if top_n:
        data = data[:top_n]
    return dict(data)

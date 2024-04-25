#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 20:04:01 2024

@author: anthony
"""
import pandas as pd


def build_item_matrix(books):
    data = books.vector.to_list()
    df = pd.DataFrame(data, index=books.id)
    return df

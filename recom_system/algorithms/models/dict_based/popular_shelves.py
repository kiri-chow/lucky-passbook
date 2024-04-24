#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:25:04 2024

@author: anthony
"""


def build_item_profile(books):
    series = books.popular_shelves
    series.index = books.id
    return series
    return dict(zip(books.id, series))

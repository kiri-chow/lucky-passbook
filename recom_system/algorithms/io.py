#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 00:13:23 2024

@author: anthony
"""
from collections.abc import Iterable
import pandas as pd
from sqlalchemy import select, func
from recom_system.db import Ratings, Books, Users, engine


def get_ratings(uidx=None, bidx=None):
    "return ratings dataframe"
    query = select(Ratings)
    if uidx is not None:
        if isinstance(uidx, Iterable):
            query = query.where(Ratings.user_id.in_(uidx))
        else:
            query = query.where(Ratings.user_id == uidx)
    if bidx is not None:
        if isinstance(bidx, Iterable):
            query = query.where(Ratings.book_id.in_(bidx))
        else:
            query = query.where(Ratings.book_id == bidx)
    with engine.begin() as con:
        df = pd.read_sql(query, con)
    return df


def get_books(bidx=None):
    "return books dataframe"
    query = select(Books)
    if bidx is not None:
        if isinstance(bidx, Iterable):
            query = query.where(Books.id.in_(bidx))
        else:
            query = query.where(Books.id == bidx)
    with engine.begin() as con:
        df = pd.read_sql(query, con)
    return df


def get_users(uidx=None):
    query = select(Users)
    if uidx is not None:
        if isinstance(uidx, Iterable):
            query = query.where(Users.id.in_(uidx))
        else:
            query = query.where(Users.id == uidx)
    with engine.begin() as con:
        df = pd.read_sql(query, con)
    return df


def get_books_by_rank(bidx=None):
    query = select(Ratings.book_id, func.sum(
        Ratings.rating), func.count(Ratings.rating))
    if bidx is not None:
        if isinstance(bidx, Iterable):
            query = query.where(Books.id.in_(bidx))
        else:
            query = query.where(Books.id == bidx)
    query = query.group_by(Ratings.book_id)

    with engine.begin() as con:
        result = con.execute(query).all()
    books = sorted(result, key=lambda x: (x[1] / x[2] > 3, x[2]), reverse=True)
    return list(map(lambda x: x[0], books))

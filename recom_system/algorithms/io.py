#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 00:13:23 2024

@author: anthony
"""
import pandas as pd
from sqlalchemy import select
from recom_system.db import Ratings, Books, engine


def get_user_ratings(uidx):
    query = select(Ratings).where(Ratings.user_id in uidx)
    with engine.connect() as con:
        df = pd.read_sql(query, con)
    return df


def get_book_data(bidx):
    query = select(Books).where(Books.id in bidx)
    with engine.connect() as con:
        df = pd.read_sql(query, con)
    return df

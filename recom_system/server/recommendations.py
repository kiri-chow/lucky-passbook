#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:09:34 2024

@author: anthony
"""
import os
import logging
import numpy as np
from sqlalchemy import select, func, and_, update
from sentence_transformers import SentenceTransformer
from recom_system.db import Users, Books, Ratings, engine, get_columns
from recom_system.algorithms.io import get_books
from recom_system.algorithms.preprocessing import get_ratings_datasets
from recom_system.algorithms.models.vector_based.sentence_embeddings import (
    build_item_matrix)
from recom_system.algorithms.models.hybird.knn_via_content import KNNViaContent
from recom_system.algorithms.models.hybird.svd_nn import SvdNCF
from surprise import SVDpp


# setup props
LLM = SentenceTransformer('all-MiniLM-L6-v2')
DIM = 384
COLUMNS = get_columns(Books)[:-1]
BOOKS = get_books()
BOOKS_PROFILE = build_item_matrix(BOOKS)
BOOKS_INDEX = BOOKS_PROFILE.index.values
BOOKS_MATRIX = BOOKS_PROFILE.values
SVD_NCF_PATH = os.path.abspath(
    os.path.join(
        os.path.split(__file__)[0],
        '..', '..', 'instance', 'SvdNCF'
    ))

# setup recommender algos
logging.info("Loading Data...")
TRAINSET, _ = get_ratings_datasets(train_size=1)
TRAINSET = TRAINSET.build_full_trainset()
logging.info("Loading SVD...")
ALGO_SVD = SVDpp(n_factors=10, n_epochs=20).fit(TRAINSET)
logging.info("Loading NCF after SVD...")
try:
    ALGO_SVD_NCF = SvdNCF.load(SVD_NCF_PATH)
except:
    logging.info("Training NCF after SVD...")
    ALGO_SVD_NCF = SvdNCF(fit_and_train=True).fit(TRAINSET)
    ALGO_SVD_NCF.save(SVD_NCF_PATH)
logging.info("Loading KNN via Content...")
ALGO_KNN_CONTENT = KNNViaContent(BOOKS_PROFILE).fit(TRAINSET)


def update_user_profile():
    rowcount = 0
    with engine.begin() as conn:
        for uid, vector in enumerate(ALGO_KNN_CONTENT.user_profiles_):
            user_id = TRAINSET.to_raw_uid(uid)
            vector = vector.tolist()
            result = conn.execute(
                update(Users).where(Users.id == user_id).values(vector=vector))
            rowcount += result.rowcount
    logging.info(f'Updated {rowcount} users\' vectors')


# update user profile
update_user_profile()


def recom_by_last_like(user_id):
    # get user last rating
    with engine.begin() as conn:
        rated = conn.execute(
            select(
                Ratings.book_id,
            ).where(and_(
                Ratings.user_id == user_id,
                Ratings.rating > 3,
            )).order_by(Ratings.modified_at.desc())
        ).all()

    if rated is None:
        return []

    rated = [x[0] for x in rated]
    last_book = rated[0]
    book_vector = BOOKS_PROFILE.loc[last_book].values

    df_books = BOOKS_PROFILE.drop(index=rated)
    books_matrix = df_books.values
    book_idx = df_books.index

    # return top 20 results
    cosine = book_vector.dot(books_matrix.T)
    index = np.argsort(cosine)[::-1][:20]
    book_idx = book_idx[index].tolist()

    books = _get_books_by_idx(book_idx)

    # add confidence
    cosine = cosine[index].tolist()
    for cos, book in zip(cosine, books):
        book['conf'] = f'{cos:.0%}'
    return books


def recom_by_knn_content(user_id):
    # get user ratings
    with engine.begin() as conn:
        ratings = conn.execute(
            select(
                Ratings.book_id, Ratings.rating,
            ).where(and_(
                Ratings.user_id == user_id,
                Ratings.rating > 3,
            ))
        ).all()
    if len(ratings) == 0:
        return []

    # conver item id
    ratings_inner = [(TRAINSET.to_inner_iid(i), r) for i, r in ratings]

    # filter items
    rated = _get_rated_items(user_id)
    all_items = []
    inner_items = []
    for iid in range(TRAINSET.n_items):
        riid = TRAINSET.to_raw_iid(iid)
        if riid in rated:
            continue
        all_items.append(riid)
        inner_items.append(iid)

    # pred the items
    scores = ALGO_KNN_CONTENT.estimate_by_ratings(ratings_inner, inner_items)

    # get top 10
    book_idx, scores = zip(*sorted(zip(all_items, scores),
                                   key=lambda x: -x[1])[:10])
    books = _get_books_by_idx(book_idx)

    # add confidence
    scores = np.clip(scores, 0, 5)
    for score, book in zip(scores, books):
        book['conf'] = f'{score/5:.0%}'
    return books


def recom_by_svd_ncf(user_id):
    try:
        uid = TRAINSET.to_inner_uid(user_id)
    # new user can not apply cf
    except ValueError:
        return []

    # filter out rated items
    rated = _get_rated_items(user_id)

    # use SVDpp to filter items
    relevant_items = []
    for iid in range(TRAINSET.n_items):
        riid = TRAINSET.to_raw_iid(iid)
        if riid in rated:
            continue
        pred = ALGO_SVD.estimate(uid, iid)

        # positive items
        if pred > 3:
            relevant_items.append(iid)

    # use Svd NCF for sorting
    scored_items = []
    for iid in relevant_items:
        pred = ALGO_SVD_NCF.estimate(uid, iid)
        scored_items.append((TRAINSET.to_raw_iid(iid), pred))

    # recommend top 10
    book_idx, scores = zip(*sorted(scored_items, key=lambda x: -x[1])[:10])
    books = _get_books_by_idx(book_idx)

    # add confidence
    scores = np.clip(scores, 0, 5)
    for score, book in zip(scores, books):
        book['conf'] = f'{score/5:.0%}'
    return books


def search_by_recom(user_id, keywords):
    """
    return result sorted by recommendation

    """
    with engine.begin() as conn:
        user = conn.execute(
            select(Users.vector).where(Users.id == user_id)
        ).one()[0]

    if user is None:
        user = np.zeros(DIM)
    else:
        user = np.array(user)
    keyword_vector = LLM.encode([keywords])[0]
    search_vector = keyword_vector + user

    norm = np.linalg.norm(search_vector)
    if norm > 0:
        search_vector /= norm
    else:
        return []

    books_matrix = BOOKS_MATRIX
    book_idx = BOOKS_INDEX

    # return top 20 results
    cosine = search_vector.dot(books_matrix.T)
    index = np.argsort(cosine)[::-1][:20]
    book_idx = book_idx[index].tolist()

    books = _get_books_by_idx(book_idx)

    # add confidence
    cosine = cosine[index].tolist()
    for cos, book in zip(cosine, books):
        book['conf'] = f'{cos:.0%}'
    return books


def _get_rated_items(user_id):
    # filter out rated items
    with engine.begin() as conn:
        rated = conn.execute(
            select(Ratings.book_id).where(Ratings.user_id == user_id)
        ).all()
    rated = set([x[0] for x in rated])
    return rated


def _get_books_by_idx(book_idx):
    with engine.begin() as conn:
        data = conn.execute(select(Books).where(Books.id.in_(book_idx))).all()

    data = sorted(data, key=lambda x: book_idx.index(x[0]))
    ratings = _get_book_ratings(book_idx)
    books = _merge_books_ratings(data, ratings)
    return books


def _merge_books_ratings(data, ratings):
    books = []
    for row in data:
        book = dict(zip(COLUMNS, row))
        books.append(book)
        try:
            r_sum, r_count = ratings[book['id']]
            book['rSum'] = r_sum
            book['rCount'] = r_count
        except KeyError:
            pass
    return books


def _get_book_ratings(book_idx):
    with engine.begin() as conn:
        ratings = conn.execute(
            select(
                Ratings.book_id,
                func.sum(Ratings.rating),
                func.count(Ratings.rating),
            ).where(
                Ratings.book_id.in_(book_idx)
            ).group_by(Ratings.book_id)).all()
    ratings = {x: (y, z) for x, y, z in ratings}
    return ratings

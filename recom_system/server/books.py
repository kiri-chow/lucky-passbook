#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 00:20:33 2024

@author: anthony
"""
import traceback
import re
from flask import Blueprint, request, jsonify
from sqlalchemy import select, func, or_
from recom_system.db import engine, Books, Ratings, get_columns
from recom_system.server.recommendations import (
    _get_book_ratings, _merge_books_ratings, 
    search_by_recom, recom_by_svd_ncf, recom_by_knn_content,
    recom_by_last_like, recom_by_user_profile,
)


REG_COMMA = re.compile(r'\,\s?')
REG_EQUAL = re.compile(r'\s?\=\s?')
COLUMNS = get_columns(Books)[:-1]


bp = Blueprint('books', __name__, url_prefix='/books')


@bp.route('/', methods=['GET'])
def get_books():
    "get books"
    page = int(request.args.get('page', 1))
    per_page = min(int(request.args.get('perPage', 20)), 100)
    sorted_by = request.args.get('sortedBy', 'date')
    keywords = request.args.get("keyword", "").lower()
    scope = request.args.get("scope", "all").lower()
    user_id = int(request.args.get("userId"))

    try:
        # search by recommendation
        if scope == 'recommend':
            books = search_by_recom(user_id, keywords)
            total = 0

        # normal search
        else:
            books, total = _search(keywords, scope, page, per_page, sorted_by)
    except BaseException as err:
        traceback.print_exc()
        return jsonify({"message": err.args[0]}), 400

    result = {
        "page": page,
        "perPage": per_page,
        "total": total,
        "data": books
    }

    return jsonify(result), 200


@bp.route('/<book_id>', methods=['GET'])
def get_one_book(book_id):
    "return a book's infomation"
    book_id = int(book_id)
    query = select(*[getattr(Books, name) for name in COLUMNS]
                   ).where(Books.id == book_id)
    with engine.begin() as conn:
        try:
            book = conn.execute(query).one()
            r_sum, r_count = conn.execute(
                select(func.sum(Ratings.rating), func.count(Ratings.rating)
                       ).where(Ratings.book_id == book_id)).one()
        except BaseException as err:
            traceback.print_exc()
            return jsonify({"message": err.args[0]}), 400
    data = dict(zip(COLUMNS, book))
    if r_count > 0:
        data['rSum'] = r_sum
        data['rCount'] = r_count
    return jsonify(data), 200


@bp.route('/recommend/<user_id>')
def get_recommendation(user_id):
    "get recommendation list for the user"
    # check recommender name
    method = request.args.get('method')
    user_id = int(user_id)

    try:
        if method == 'svd_ncf':
            books = recom_by_svd_ncf(user_id)
        elif method == 'knn_content':
            books = recom_by_knn_content(user_id)
        elif method == 'last_like':
            books = recom_by_last_like(user_id)
        elif method == 'cold_start':
            books = recom_by_user_profile(user_id)
        else:
            raise ValueError("Recommendation TBC")
    except BaseException as err:
        traceback.print_exc()
        return jsonify({"message": err.args[0]}), 400
    
    return books


def _search(keywords, scope, page, per_page, sorted_by):
    # compute offset
    offset = (page - 1) * per_page

    # search condition
    conditions = []
    if scope == 'all':
        scope = ['title', 'description']
    else:
        scope = [scope]
    if keywords:
        for key in scope:
            conditions.append(getattr(Books, key).icontains(keywords))

    query = select(*[getattr(Books, name) for name in COLUMNS])
    count = select(func.count(Books.id))

    if sorted_by == 'date':
        query = query.order_by(Books.publication_year.desc())

    if conditions:
        query = query.where(or_(*conditions))
        count = count.where(or_(*conditions))

    query = query.offset(offset).limit(per_page)
    with engine.begin() as conn:
        total = conn.execute(count).one()[0]
        data = conn.execute(query).all()

    book_idx = [row[0] for row in data]
    ratings = _get_book_ratings(book_idx)
    books = _merge_books_ratings(data, ratings)
    return books, total

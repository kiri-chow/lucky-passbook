#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 00:20:33 2024

@author: anthony
"""
import re
from flask import Blueprint, request, jsonify
from sqlalchemy import select, func, or_
from recom_system.db import engine, Books, get_columns


REG_COMMA = re.compile(r'\,\s?')
REG_EQUAL = re.compile(r'\s?\=\s?')
COLUMNS = get_columns(Books)[:-1]


bp = Blueprint('books', __name__, url_prefix='/books')


@bp.route('/', methods=['GET'])
def get_books():
    "get books"
    page = int(request.args.get('page', 1))
    per_page = min(int(request.args.get('perPage', 20)), 100)
    offset = (page - 1) * per_page

    # add search condition
    conditions = []
    keywords = request.args.get("keyword", "").lower()
    slope = request.args.get("scope", "all").lower()
    if slope == 'all':
        slope = ['title', 'description']
    else:
        slope = [slope]
    if keywords:
        for key in slope:
            conditions.append(getattr(Books, key).icontains(keywords))

    query = select(*[getattr(Books, name) for name in COLUMNS])
    count = select(func.count(Books.id))
    if conditions:
        query = query.where(or_(*conditions))
        count = count.where(or_(*conditions))
    query = query.offset(offset).limit(per_page)
    with engine.begin() as conn:
        try:
            total = conn.execute(count).one()[0]
            data = conn.execute(query).all()
        except BaseException as err:
            return jsonify({"message": err.args[0]}), 400

    result = {
        "page": page,
        "perPage": per_page,
        "total": total,
        "data": [dict(zip(COLUMNS, row)) for row in data]
    }

    return jsonify(result), 200


@bp.route('/recommend/<user_id>')
def get_recommendation(user_id):
    "get recommendation list for the user"
    # check recommender name
    method = request.args.get('method')
    if not method:
        method = "default"

    # TODO: implement the actual methods
    return jsonify({"message": "API TBC"}), 500

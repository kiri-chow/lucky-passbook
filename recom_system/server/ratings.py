#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 23:48:33 2024

@author: anthony
"""
from flask import Blueprint, request, jsonify
from sqlalchemy import select, insert, update, delete
from recom_system.db import engine, Ratings


COLUMNS = ['id', 'userId', 'bookId', 'rating']
bp = Blueprint('ratings', __name__, url_prefix='/ratings')


@bp.route('/', methods=['GET'])
def get_ratings():
    "get ratings"
    # check user id or item id
    user_id = request.args.get('userId')
    item_id = request.args.get('bookId')

    if user_id is None and item_id is None:
        return jsonify({"message": "requiring either userId or bookId"}), 400

    
    query = select(Ratings.id, Ratings.user_id,
                   Ratings.book_id, Ratings.rating)
    if user_id is not None:
        query = query.where(Ratings.user_id == user_id)
    if item_id is not None:
        query = query.where(Ratings.book_id == item_id)
    with engine.begin() as conn:
        cursor = conn.execute(query)
        result = [dict(zip(COLUMNS, i)) for i in cursor]

    return jsonify(result), 200


@bp.route('/<rid>', methods=['GET'])
def get_one_rating(rid):
    "get one rating"
    query = select(Ratings.user_id, Ratings.book_id, Ratings.rating
                   ).where(Ratings.id == rid)
    try:
        with engine.begin() as conn:
            data = conn.execute(query).one()
    except BaseException as err:
        return jsonify({"message": err.args[0]}), 404
    return jsonify(dict(zip(COLUMNS, data))), 200


@bp.route('/', methods=['POST'])
def create_ratings():
    "create a new rating"
    # check user id, item id, and rating
    user_id = request.json.get('userId')
    if user_id is None:
        return jsonify({"message": "requiring user_id"}), 400
    book_id = request.json.get('bookId')
    if book_id is None:
        return jsonify({"message": "requiring item_id"}), 400
    rating = request.json.get('rating')
    if rating is None:
        return jsonify({"message": "requiring rating"}), 400

    # insert
    query = insert(Ratings).values(
        user_id=user_id, book_id=book_id, rating=rating)
    try:
        with engine.begin() as conn:
            result = conn.execute(query)
        if result.rowcount == 1:
            return jsonify({
                "message": "success",
                "id": result.inserted_primary_key[0],
            }), 200
    except BaseException as err:
        return jsonify({"message": err.args[0]}), 400
    return jsonify({"message": "unknown error"}), 500


@bp.route('/<rid>', methods=['PUT'])
def update_ratings(rid):
    "modify a rating"
    rating = request.json.get('rating')
    if rating is None:
        return jsonify({"message": "requiring rating"}), 400

    # update
    query = update(Ratings).where(Ratings.id == rid).values(rating=rating)
    try:
        with engine.begin() as conn:
            result = conn.execute(query)
        if result.rowcount == 1:
            return jsonify({"message": "success", "id": rid}), 200
    except BaseException as err:
        return jsonify({"message": err.args[0]}), 400
    return jsonify({"message": "unknown error"}), 500


@bp.route('/<rid>', methods=['DELETE'])
def delete_ratings(rid):
    # delete
    query = delete(Ratings).where(Ratings.id == rid)
    try:
        with engine.begin() as conn:
            result = conn.execute(query)
        if result.rowcount == 1:
            return jsonify({"message": "success"}), 200
    except BaseException as err:
        return jsonify({"message": err.args[0]}), 400
    return jsonify({"message": "unknown error"}), 500

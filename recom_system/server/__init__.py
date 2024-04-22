#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:59:59 2024

@author: anthony
"""
import os
from flask import Flask, request, jsonify
from sqlalchemy import select, insert
from recom_system.db import engine, Users


STATIC_PATH = os.path.join(os.path.split(__file__)[0], 'static')


def create_app():
    app = Flask(
        'lucky-pass-book',
        static_url_path='/',
        static_folder=STATIC_PATH,
    )

    @app.route('/api/login', methods=['POST'])
    def login():
        "Log a user in, register if the user does not exist"
        username = request.json.get('username')
        if not username:
            return jsonify({"message": "requiring username"}), 400

        # get user id
        user_id = None
        query = select(Users.id).where(Users.name == username)
        with engine.begin() as conn:
            try:
                user_id = conn.execute(query).one()
            except:
                pass
            else:
                user_id = user_id[0]

        # register
        if user_id is None:
            with engine.begin() as conn:
                query = insert(Users).values(name=username)
                result = conn.execute(query)
            user_id = result.inserted_primary_key[0]

        return jsonify({'id': user_id}), 200


    from . import ratings
    app.register_blueprint(ratings.bp, url_prefix='/api/ratings')

    from . import books
    app.register_blueprint(books.bp, url_prefix='/api/books')

    return app

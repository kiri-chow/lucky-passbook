#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:17:30 2024

@author: anthony
"""
import os
import sqlalchemy
from sqlalchemy.orm import declarative_base
from sqlalchemy import (
    Column, Integer, String, Boolean, JSON, Text,
    ForeignKey, UniqueConstraint
)


URL = os.getenv('RS_SQLURL')

# test env
if not URL:
    URL = 'sqlite:///' + os.path.abspath(
        os.path.join(
            os.path.split(__file__)[0], '..', 'instance', 'data.db'
        )
    )

engine = sqlalchemy.create_engine(URL)


Base = declarative_base()


def get_columns(table):
    return [col.name for col in table.__table__.columns]


class Users(Base):
    "The users table"

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)

    name = Column(String)

    def __repr__(self):
        return f'<User(name={self.name})>'


class Ratings(Base):
    "The ratings table"

    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True)

    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    book_id = Column(Integer, ForeignKey('books.id'), nullable=False)

    is_read = Column(Boolean, nullable=False, default=True)

    rating = Column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint('user_id', 'book_id', name='uid_bid'),
    )

    def __repr__(self):
        return f'<Rating(user={self.user_id}, book={self.book_id}, rating={self.rating})>'


class Books(Base):
    "the books table"

    __tablename__ = "books"

    id = Column(Integer, primary_key=True)

    title = Column(String)

    popular_shelves = Column(JSON)

    series = Column(JSON)

    authors = Column(JSON)

    description = Column(Text)

    image_url = Column(String)

    url = Column(String)

    publication_year = Column(Integer)

    is_ebook = Column(Boolean)

    vector = Column(JSON)

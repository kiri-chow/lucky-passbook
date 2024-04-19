#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:59:59 2024

@author: anthony
"""
import os
from flask import Flask


STATIC_PATH = os.path.join(os.path.split(__file__)[0], 'static')


app = Flask(
    'lucky-pass-book',
    static_url_path='/',
    static_folder=STATIC_PATH,
)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 23:06:37 2024

@author: anthony
"""
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer


try:
    NLP = spacy.load("en_core_web_md")
except OSError as exc:
    raise RuntimeError(
        "Please run 'python -m spacy download en_core_web_md' "
        "to download the nlp model."
    ) from exc

LLM = SentenceTransformer('all-MiniLM-L6-v2')


def vectorize_book(book):
    "vectorize the book by title and description"
    # split into sentences
    sents = []
    doc = NLP(book.description)
    for sent in doc.sents:
        # only extract from the first 5 sentences from description
        if len(sents) > 5:
            break
        if len(sent) < 2:
            continue
        sents.append(sent.text)

    # vectorize sentences
    if len(book.title.split(' ')) < 2:
        vectors = LLM.encode(sents)
    else:
        vectors = LLM.encode([book.title] + sents)
        vectors[0] *= 2

    vector = vectors.sum(0)

    # normalization
    vector /= np.linalg.norm(vector)
    return vector

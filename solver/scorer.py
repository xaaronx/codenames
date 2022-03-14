from __future__ import annotations

from operator import itemgetter

import numpy as np

from solver.utils import get_top_n_sorted


class Scorer:
    def __init__(self, guesses: list, n: int):
        self.guesses = guesses
        self.n = n

    @staticmethod
    def score_single(guess: Guess):
        return guess.similarity_score * np.log(guess.num_words_linked)

    def top_n(self, metric):
        scores = np.array([guess.__getattribute__(metric) for guess in self.guesses])
        return get_top_n_sorted(scores, self.n)

    def top_n_words(self, metric):
        ixs = self.top_n(metric)
        relevant_guesses = list(itemgetter(*ixs)(self.guesses))
        return [guess.__getattribute__('clue') for guess in relevant_guesses]


class Guess:
    def __init__(self, clue: str, similarity_score: float, linked_words: list):
        self.clue = clue
        self.similarity_score = similarity_score
        self.linked_words = linked_words
        self.num_words_linked = len(self.linked_words)

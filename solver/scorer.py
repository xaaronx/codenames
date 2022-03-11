from __future__ import annotations

import numpy as np


class Scorer:
    def __init__(self):
        pass

    def score_single(self, guess: Guess):
        return guess.similarity_score * np.log(guess.num_words_linked)

    def top_n(self, guesses: list):
        pass


class Guess:
    def __init__(self, clue: str, similarity_score: float, linked_words: list):
        self.clue = clue
        self.similarity_score = similarity_score
        self.linked_words = linked_words
        self.num_words_linked = len(self.linked_words)

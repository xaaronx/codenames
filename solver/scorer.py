from __future__ import annotations

from dataclasses import dataclass, field
from operator import itemgetter

import numpy as np
from scipy.spatial.distance import cosine

from solver.utils import get_top_n_sorted


class Scorer:
    def __init__(self, guesses: list, embeddings: dict, n: int, threshold: float, metric: str = "similarity_score"):
        self.guesses = guesses
        self.embeddings = embeddings
        self.n = n
        self.metric = metric
        self.threshold = threshold

    def _filter_illegal(self):
        return [
            guess for guess in self.guesses
            if all(map(lambda word: True if word not in guess.clue else False, guess.linked_words))
        ]

    def _score_single(self, guess: Guess) -> Guess:
        """Takes metric of choice (from class) and multiplies by log of number of words linked.

        :param guess: Single Guess object
        :return: Updated guess object
        """
        guess.score = guess.__getattribute__(self.metric) * guess.num_words_linked
        return guess

    def _score_single_double_check(self, guess: Guess) -> Guess:
        connected = [1 - cosine(self.embeddings.get(word, 0), self.embeddings.get(guess.clue)) > self.threshold
                     for word in guess.linked_words]

        if all(connected):
            guess.score = guess.__getattribute__(self.metric)
            return guess
        else:
            guess.score = 0
            return guess

    def _top_n(self) -> np.array:
        """Scores guesses, gets scores from guess object and then find indices of top n

        :return: Indices of top scores
        """
        scored_guesses = list(map(self._score_single_double_check, self.guesses))
        scores = np.array(list(map(lambda guess: guess.score, scored_guesses)))
        top_ixs = get_top_n_sorted(scores, self.n)
        return top_ixs

    def top_n_guesses(self) -> list:
        """Gets top n guess objects using _top_n method

        :return: list of Guess objects that score highest.
        """
        self.guesses = self._filter_illegal()
        ixs = self._top_n()
        top_guesses = list(itemgetter(*ixs)(self.guesses))
        return top_guesses


@dataclass
class Guess:
    clue: str
    similarity_score: float
    linked_words: list
    num_words_linked: int = field(init=False)

    def __post_init__(self):
        self.num_words_linked = self.get_num_words_linked()

    def get_num_words_linked(self) -> int:
        return len(self.linked_words)

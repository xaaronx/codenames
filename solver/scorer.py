from __future__ import annotations

from dataclasses import dataclass, field
from operator import itemgetter

import numpy as np

from solver.utils import get_top_n_sorted


class Scorer:
    def __init__(self, guesses: list, n: int, metric: str = "similarity_score"):
        self.guesses = guesses
        self.n = n
        self.metric = metric

    def score_single(self, guess: Guess) -> Guess:
        """Takes metric of choice (from class) and multiplies by log of number of words linked.

        :param guess: Single Guess object
        :return: Updated guess object
        """
        guess.score = guess.__getattribute__(self.metric) * np.cbrt(guess.num_words_linked)
        return guess

    def _top_n(self) -> np.array:
        """Scores guesses, gets scores from guess object and then find indices of top n

        :return: Indices of top scores
        """
        scored_guesses = list(map(self.score_single, self.guesses))
        scores = np.array(list(map(lambda guess: guess.score, scored_guesses)))
        top_ixs = get_top_n_sorted(scores, self.n)
        return top_ixs

    def top_n_guesses(self) -> list:
        """Gets top n guess objects using _top_n method

        :return: list of Guess objects that score highest.
        """
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


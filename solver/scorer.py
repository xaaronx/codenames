from __future__ import annotations

from operator import itemgetter

import numpy as np
from scipy.spatial.distance import cosine

from solver.guess import Guess
from solver.utils import get_top_n_sorted


class EmbeddingScorer:
    def __init__(self, guesses: list, embeddings: dict, words_to_avoid: list, n: int,
                 threshold: float, metric: str = "similarity_score", incorrect_words_threshold: float = 2):
        self.guesses = guesses
        self.embeddings = embeddings
        self.words_to_avoid = words_to_avoid
        self.n = n
        self.metric = metric
        self.threshold = threshold
        self.incorrect_words_threshold = incorrect_words_threshold

    def _score_single(self, guess: Guess) -> Guess:
        """Takes metric of choice (from class) and multiplies by log of number of words linked.

        :param guess: Single Guess object
        :return: Updated guess object
        """
        guess.score = getattr(guess, self.metric) * guess.num_words_linked
        return guess

    @staticmethod
    def _check_all_legal(guess: Guess) -> Guess:
        """Check if clue is legal (e.g. fish -> catfish)

        :return: Guess object
        """
        if all(map(lambda word: True if (word not in guess.clue and guess.clue not in word) else False,
                   guess.linked_words)):
            return guess

    def _check_all_connected(self, guess: Guess) -> Guess:
        """Check if all words are sufficiently similar to the clue based on self.threshold and cosine similarity.

        :param guess: Guess object
        :return: Guess object
        """
        if all(1 - cosine(self.embeddings.get(word, 0),
                          self.embeddings.get(guess.clue)
                          ) > self.threshold
               for word in guess.linked_words):
            return guess

    def _check_incorrect_matches(self, guess: Guess) -> Guess:
        """Checks against opponents, neutral and bomb words (all contained in self.words_to_avoid). Akin to
        _check_all_connected, finds cosine similarity between clue and aforementioned anti-words and if any
        over threshold returns None. Threshold is set to be more liberal

        :param guess: Guess object
        :return: Guess object
        """
        if self.words_to_avoid:
            if all(1 - cosine(self.embeddings.get(word, 0),
                              self.embeddings.get(guess.clue)
                              ) < (self.threshold * self.incorrect_words_threshold
                   ) for word in self.words_to_avoid):
                return guess
        else:
            return guess

    def _preprocess(self):
        """Runs preprocessing steps to filter out bad guesses. Currently, check if all words sufficiently connected to
        clue, if all clues are legal and if all words are sufficiently disimilar to incorrect matches.
        """
        preprocessing_steps = [
            self._check_all_connected,
            self._check_incorrect_matches,
            self._check_all_legal
        ]

        for step in preprocessing_steps:
            self.guesses = list(map(step, self.guesses))
            self.guesses = list(filter(None, self.guesses))

    def _top_n(self) -> np.array:
        """Scores guesses, gets scores from guess object and then find indices of top n

        :return: Indices of top scores
        """
        scored_guesses = list(map(self._score_single, self.guesses))
        scores = np.array(list(map(lambda guess: guess.score, scored_guesses)))
        top_ixs = get_top_n_sorted(scores, self.n)
        return top_ixs

    def top_n_guesses(self) -> list:
        """Gets top n guess objects using _top_n method

        :return: list of Guess objects that score highest.
        """
        self._preprocess()
        ixs = self._top_n()
        top_guesses = list(itemgetter(*ixs)(self.guesses))
        return top_guesses

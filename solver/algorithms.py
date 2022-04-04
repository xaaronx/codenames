import itertools
import logging
from operator import itemgetter

import numpy as np

from solver.scorer import Guess, EmbeddingScorer
from solver.utils import remove_keys_from_dict, get_top_n_sorted


class CodeNamesSolverAlgorithm:
    def __init__(self):
        pass

    @staticmethod
    def cosine_vectorized(array1, array2):
        sumyy = (array2 ** 2).sum(1)
        sumxx = (array1 ** 2).sum(1, keepdims=1)
        sumxy = array1.dot(array2.T)
        return (sumxy / np.sqrt(sumxx)) / np.sqrt(sumyy)

    @staticmethod
    def nearest_neighbor_search(base_vector, target_array, n):
        # Find distance
        similarities = np.dot(base_vector, target_array)
        # Get top n matches
        indices = get_top_n_sorted(similarities, n)
        return indices, similarities

    @staticmethod
    def get_word_combinations(words_to_hit: list) -> list:
        return list(itertools.chain(*map(lambda x: itertools.combinations(words_to_hit, x),
                                         range(1, len(words_to_hit) + 1))))


class NearestNeighborSum(CodeNamesSolverAlgorithm):
    def __init__(self, model: dict, words_to_hit: list, n: int, threshold: float, words_to_avoid: list = []):
        super().__init__()
        self.model = model
        self.words_to_hit = words_to_hit
        self.words_to_avoid = words_to_avoid
        self.n = n
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def _compute(self, words: list) -> list:
        """Computes nearest neighbors (best guesses) for a single combination of words. Uses sum of embedding vectors
        of words to hit and finds the nearest word to this summed vector.

        :param words: Words to connect.
        :return: List of best guesses for a single combination of words.
        """

        # Fetch embeddings for words of relevance
        embeddings_of_words_to_hit = np.array([self.model.get(word, np.array) for word in list(words)])
        # Get sum of fetched embeddings
        target_vector = np.sum(embeddings_of_words_to_hit, axis=0)
        # Remove words_to_hit from potential matches
        potential_match_embeddings = remove_keys_from_dict(self.model, words)
        # Convert to array
        embeddings_as_array = np.array(list(potential_match_embeddings.values()))
        # Find nearest
        indices, similarities = self.nearest_neighbor_search(target_vector, embeddings_as_array.T, self.n * 10)
        # Index matches against original list
        matched_words = itemgetter(*indices)(list(potential_match_embeddings.keys()))
        # Fetch also the numerical similarities
        sims = similarities[indices]
        return list(zip(matched_words, sims))

    def solve(self) -> list:
        guesses = []
        words_combinations = self.get_word_combinations(self.words_to_hit)
        for words in words_combinations:
            for solution in self._compute(words):
                clue, similarity = solution
                guess = Guess(clue, similarity, words)
                guesses.append(guess)

        return EmbeddingScorer(guesses=guesses,
                               embeddings=self.model,
                               words_to_avoid=self.words_to_avoid,
                               n=self.n,
                               threshold=self.threshold
                               ).top_n_guesses()


class BestAverageAngle(CodeNamesSolverAlgorithm):
    def __init__(self, model: dict, words_to_hit: list, n: int, threshold: float, words_to_avoid: list = []):
        super().__init__()
        self.model = model
        self.words_to_hit = words_to_hit
        self.words_to_avoid = words_to_avoid
        self.n = n
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def _compute(self, words: list) -> list:
        # Fetch embeddings for words of relevance
        embeddings_of_words_to_hit = np.array([self.model.get(word, np.array) for word in list(words)])
        # Remove words_to_hit from potential matches
        potential_match_embeddings = remove_keys_from_dict(self.model, words)
        # And convert to array
        potential_match_embeddings_array = np.vstack(list(potential_match_embeddings.values()))
        # Calculate cosine similarities between each candidate and each word to hit
        sims = self.cosine_vectorized(potential_match_embeddings_array, embeddings_of_words_to_hit)
        # Average to get mean similarity of each candidate to all words to hit
        mean_sims = sims.T.mean(axis=0)
        # Get top n
        indices = get_top_n_sorted(mean_sims, self.n * 10)
        # Index matches against original list
        matched_words = itemgetter(*indices)(list(potential_match_embeddings.keys()))
        # Fetch also the numerical similarities
        sims = mean_sims[indices]
        return list(zip(matched_words, sims))

    def solve(self) -> list:
        guesses = []
        words_combinations = self.get_word_combinations(self.words_to_hit)
        for words in words_combinations:
            for solution in self._compute(words):
                clue, similarity = solution
                guess = Guess(clue, similarity, words)
                guesses.append(guess)

        return EmbeddingScorer(guesses=guesses,
                               embeddings=self.model,
                               words_to_avoid=self.words_to_avoid,
                               n=self.n,
                               threshold=self.threshold
                               ).top_n_guesses()

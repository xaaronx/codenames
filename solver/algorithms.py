import itertools
import logging
from operator import itemgetter

import numpy as np

from solver.distance import Cosine, DotProduct
from solver.scorer import Guess, EmbeddingScorer
from solver.utils import remove_keys_from_dict, get_top_n_sorted


class CodeNamesSolverAlgorithm:
    def __init__(self, model: dict, threshold: float, distance_metric=Cosine, search_space_multiplier: int = 10):
        self.model = model
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.search_space_multiplier = search_space_multiplier
        self.logger = logging.getLogger(__name__)

    def solve(self, words_to_hit: list, words_to_avoid: list = [], n: int = 10):
        guesses = []
        words_combinations = self._get_word_combinations(words_to_hit)
        for words in words_combinations:
            for solution in self._compute(words, n):
                clue, similarity = solution
                guess = Guess(clue, similarity, words)
                guesses.append(guess)

        return self._get_top_guesses(guesses, words_to_avoid, n)

    @staticmethod
    def _get_word_combinations(words_to_hit: list) -> list:
        return list(itertools.chain(*map(lambda x: itertools.combinations(words_to_hit, x),
                                         range(1, len(words_to_hit) + 1))))

    def _compute(self, *args, **kwargs):
        raise NotImplementedError

    def _get_top_guesses(self, guesses: list, words_to_avoid: list, n: int):
        return EmbeddingScorer(guesses=guesses,
                               embeddings=self.model,
                               distance_metric=self.distance_metric,
                               words_to_avoid=words_to_avoid,
                               n=n,
                               threshold=self.threshold
                               ).top_n_guesses()


class MeanIndividualDistance(CodeNamesSolverAlgorithm):
    def __init__(self, model: dict, threshold: float, search_space_multiplier: int = 10, distance_metric=Cosine):
        super().__init__(model, threshold, distance_metric, search_space_multiplier)

    def _compute(self, words: list, n) -> list:
        # Fetch embeddings for words of relevance
        embeddings_of_words_to_hit = np.array([self.model.get(word, np.array) for word in list(words)])
        # Remove words_to_hit from potential matches
        potential_match_embeddings = remove_keys_from_dict(self.model, words)
        # And convert to array
        potential_match_embeddings_array = np.vstack(list(potential_match_embeddings.values()))
        # Calculate cosine similarities between each candidate and each word to hit
        sims = self.distance_metric(potential_match_embeddings_array, embeddings_of_words_to_hit).distance()
        # Average to get mean similarity of each candidate to all words to hit
        mean_sims = sims.T.mean(axis=0)
        # Get top n
        indices = get_top_n_sorted(mean_sims, n * self.search_space_multiplier)
        # Index matches against original list
        matched_words = itemgetter(*indices)(list(potential_match_embeddings.keys()))
        # Fetch also the numerical similarities
        sims = mean_sims[indices]
        return list(zip(matched_words, sims))


class SummedNearestNeighbour(CodeNamesSolverAlgorithm):
    def __init__(self, model: dict, threshold: float, search_space_multiplier: int = 10, distance_metric=DotProduct):
        super().__init__(model, threshold, distance_metric, search_space_multiplier)

    def _compute(self, words: list, n: int) -> list:
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
        similarities = np.squeeze(self.distance_metric(target_vector, embeddings_as_array).distance())
        # Get top n matches
        indices = get_top_n_sorted(similarities, n * self.search_space_multiplier)
        # Index matches against original list
        matched_words = itemgetter(*indices)(list(potential_match_embeddings.keys()))
        # Fetch also the numerical similarities
        sims = similarities[indices]
        return list(zip(matched_words, sims))

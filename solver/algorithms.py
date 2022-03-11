import itertools
from operator import itemgetter

import numpy as np

from solver.scorer import Guess
from solver.utils import remove_keys_from_dict, get_top_n_sorted


class CodeNamesSolverAlgorithm:
    def __init__(self):
        pass

    @staticmethod
    def nearest_neighbor_search(base_vector, target_array, n):
        # Find distance
        similarities = np.dot(base_vector, target_array)
        # Get top n matches
        indices = get_top_n_sorted(similarities, n)
        return indices, similarities

    @staticmethod
    def get_word_combinations(words_to_hit: list):
        return list(itertools.chain(*map(lambda x: itertools.combinations(words_to_hit, x),
                                         range(1, len(words_to_hit) + 1))))


class NearestNeighborMean(CodeNamesSolverAlgorithm):
    def __init__(self, embeddings: dict, words_to_hit: list, n: int):
        super().__init__()
        self.embeddings = embeddings
        self.words_to_hit = words_to_hit
        self.n = n

    def compute(self, words: list):
        # Fetch embeddings for words of relevance
        embeddings_of_words_to_hit = np.array([self.embeddings.get(word, np.array) for word in list(words)])
        # Get mean of fetched embeddings
        target_vector = np.mean(embeddings_of_words_to_hit, axis=0)
        # Remove words_to_hit from potential matches
        potential_match_embeddings = remove_keys_from_dict(self.embeddings, words)
        # Conver to array
        embeddings_as_array = np.array(list(potential_match_embeddings.values()))
        # Find nearest
        indices, similarities = self.nearest_neighbor_search(target_vector, embeddings_as_array.T, self.n)
        # Index maches against original list
        words = itemgetter(*indices)(list(potential_match_embeddings.keys()))
        # Fetch also the numerical similarities
        sims = similarities[indices]
        return list(zip(words, sims))

    def solve(self):
        guesses = []
        words_combinations = self.get_word_combinations(self.words_to_hit)
        for words in words_combinations:
            for solution in self.compute(words):
                clue, similarity = solution
                guesses.append(Guess(clue, similarity, words))

        solutions = np.array(guesses)
        print(solutions[solutions[:, 1].argsort()][:20])

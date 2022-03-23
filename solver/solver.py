import logging

import numpy as np
from tqdm import tqdm


class Solver:
    def __init__(self, words_to_hit: list, words_to_avoid: list, embeddings: dict, n: int):
        self.words_to_hit = words_to_hit
        self.words_to_avoid = words_to_avoid
        self.embeddings = embeddings
        self.n = n

    def solve(self, algorithm) -> list:
        """Takes algorithm object and gives prediction for best clues to link your words and avoid words that are not
        yours.

        :param algorithm: A solver.algorithm object that contains and solve method.
        :return: List of self.n Guess objects.
        """
        return algorithm(words_to_hit=self.words_to_hit, embeddings=self.embeddings, n=self.n).solve()


class SolverBuilder:
    def __init__(self, words_to_hit: list, words_to_avoid: list, embedding_path: str, n: int = 5):
        self.words_to_hit = words_to_hit
        self.words_to_avoid = words_to_avoid
        self.embedding_path = embedding_path
        self.n = n
        self.logger = logging.getLogger(__name__)

    def _persist_embeddings(self):
        raise NotImplementedError

    def build(self) -> Solver:
        embeddings = self._persist_embeddings()
        return Solver(words_to_hit=self.words_to_hit,
                      words_to_avoid=self.words_to_avoid,
                      embeddings=embeddings,
                      n=self.n)


class GloveSolver(SolverBuilder):
    def __init__(self, words_to_hit: list, words_to_avoid: list, embedding_path: str, n: int = 5):
        super().__init__(words_to_hit, words_to_avoid, embedding_path, n)
        self.logger = logging.getLogger(__name__)

    def _persist_embeddings(self) -> dict:
        embeddings = {}
        with open(self.embedding_path, "r") as file:
            for line in tqdm(file):
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                embeddings[word] = embedding
        self.logger.info("Glove embeddings loaded.")
        return embeddings


class AdversarialPostSpecSolver(SolverBuilder):
    def __init__(self, words_to_hit: list, words_to_avoid: list, embedding_path: str, n: int = 5):
        # See https://github.com/cambridgeltl/adversarial-postspec
        super().__init__(words_to_hit, words_to_avoid, embedding_path, n)
        self.logger = logging.getLogger(__name__)

    def _persist_embeddings(self) -> dict:
        embeddings = {}
        with open(self.embedding_path, "r") as file:
            for line in tqdm(file):
                split_line = line.split()
                word = split_line[0].split('_')
                if word[0] == 'en':
                    word = word[1]
                    embedding = np.array(split_line[1:], dtype=np.float64)
                    embeddings[word] = embedding
        self.logger.info("Glove embeddings loaded.")
        return embeddings

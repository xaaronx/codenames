import logging

import numpy as np


class Solver:
    def __init__(self, words_to_hit: list, words_to_avoid: list, embeddings: dict, n: int = 5):
        self.words_to_hit = words_to_hit
        self.words_to_avoid = words_to_avoid
        self.embeddings = embeddings
        self.n = n

    def solve(self, algorithm) -> list:
        return algorithm(words_to_hit=self.words_to_hit, embeddings=self.embeddings, n=self.n).solve()


class SolverBuilderGlove:
    def __init__(self, words_to_hit: list, words_to_avoid: list, embedding_path: str):
        self.words_to_hit = words_to_hit
        self.words_to_avoid = words_to_avoid
        self.embedding_path = embedding_path
        self.logger = logging.getLogger(__name__)

    def persist_embeddings(self) -> dict:
        embeddings = {}
        with open(self.embedding_path, "r") as file:
            for line in file:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                embeddings[word] = embedding
        self.logger.info("Glove embeddings loaded.")
        return embeddings

    def build(self) -> Solver:
        embeddings = self.persist_embeddings()
        return Solver(words_to_hit=self.words_to_hit,
                      words_to_avoid=self.words_to_avoid,
                      embeddings=embeddings)

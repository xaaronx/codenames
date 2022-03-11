import os

from solver.algorithms import NearestNeighborMean
from solver.solver import SolverBuilderGlove

if __name__ == "__main__":
    words_to_avoid = []
    words_to_hit = ["cheese", "milk", "butter", "sugar"]

    embedding_path = os.path.join("..", "data", "word_embeddings", "glove.6B.50d.txt")

    solver = SolverBuilderGlove(words_to_hit=words_to_hit,
                                words_to_avoid=words_to_avoid,
                                embedding_path=embedding_path
                                ).build()

    solver.solve(NearestNeighborMean)

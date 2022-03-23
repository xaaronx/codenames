import logging
import os

from solver.algorithms import NearestNeighborMean
from solver.solver import AdversarialPostSpecSolver

if __name__ == "__main__":
    words_to_avoid = []
    # words_to_hit = ["milk", "cheese", "yogurt", "butter"]
    words_to_hit = ["france", "germany", "america", "russia"]

    logging.getLogger().setLevel('INFO')

    # embedding_path = os.path.join("..", "data", "word_embeddings", "glove", "glove.6B.300d.txt")
    embedding_path = os.path.join("..", "data", "word_embeddings", "post-specialized embeddings", "postspec",
                                  "glove_postspec.txt")
    num_guesses = 20

    solver = AdversarialPostSpecSolver(words_to_hit=words_to_hit,
                                       words_to_avoid=words_to_avoid,
                                       embedding_path=embedding_path,
                                       n=num_guesses
                                       ).build()

    solutions = solver.solve(NearestNeighborMean)
    for i in solutions:
        print(i.clue, i.linked_words, i.score)

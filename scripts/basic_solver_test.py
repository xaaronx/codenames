import os
import time

from solver.algorithms import MeanIndividualDistance, SummedNearestNeighbour
from solver.distance import Cosine, DotProduct
from solver.solver import SolverBuilder
from solver.utils import initialise_logger, log_solutions

if __name__ == "__main__":
    logger = initialise_logger()
    # embedding_path = os.path.join("..", "data", "word_embeddings", "glove", "glove.6B.300d.txt")
    embedding_path = os.path.join("..", "data", "word_embeddings", "definitional", "embedding.txt")
    conf_path = os.path.join("..", "data", "params.csv")
    builder = SolverBuilder.with_bert(embedding_path)
    solver = builder.build(conf_path, SummedNearestNeighbour, Cosine)

    words = input("Enter some words to connect with a space in between each...")
    while words:
        print("Thinking...")
        words = words.split(' ')
        log_solutions(solver.solve(words))
        time.sleep(0.5)
        words = input("Enter some more words to connect with a space in between each or press enter to escape...")
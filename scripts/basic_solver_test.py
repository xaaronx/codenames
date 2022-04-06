import os

from solver.solver import SolverBuilder
from solver.utils import initialise_logger, log_solutions

if __name__ == "__main__":
    logger = initialise_logger()
    glove_embedding_path = os.path.join("..", "data", "word_embeddings", "glove", "glove.6B.50d.txt")
    conf_path = os.path.join("..", "data", "params.csv")
    builder = SolverBuilder.with_glove(glove_embedding_path)
    solver = builder.build(conf_path)
    log_solutions(solver.solve(["dog", "cat"]))

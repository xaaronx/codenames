import logging
import os
import random

from game.wordlist import WordListBuilder
from solver.algorithms import NearestNeighborSum
from solver.solver import AdversarialPostSpecSolver, GloveSolver


def initialise_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(message)8s')
    return logger


def run_glove_solver(algo, t):
    glove_embedding_path = os.path.join("..", "data", "word_embeddings", "glove", "glove.6B.300d.txt")
    glove_solver = GloveSolver(**CONFIG, embedding_path=glove_embedding_path, threshold=t).build()
    return glove_solver.solve(algo)


def run_postspec_solver(algo, t):
    p_path = ["..", "data", "word_embeddings", "post-specialized embeddings", "postspec", "glove_postspec.txt"]
    postspec_embedding_path = os.path.join(*p_path)
    postspec_solver = AdversarialPostSpecSolver(**CONFIG, embedding_path=postspec_embedding_path, threshold=t).build()
    return postspec_solver.solve(algo)


def log_solutions(solutions):
    dash = '-' * 80
    formatting = '{:<20s}{:<30s}{:<40}'
    for i, s in enumerate(solutions):
        if i == 0:
            LOGGER.info(dash)
            LOGGER.info(formatting.format("Clue", "Score", "Linked Words"))
            LOGGER.info(dash)
        else:
            LOGGER.info(formatting.format(s.clue, str(round(s.score, 3)), ', '.join(s.linked_words)))


if __name__ == "__main__":
    LOGGER = initialise_logger()
    n = 20
    algorithm = NearestNeighborSum

    path_to_word_list = os.path.join("..", "data", "wordlist-eng.txt")
    wordlist = WordListBuilder(path_to_word_list).build().wordlist

    words_to_avoid = []
    words_to_hit = random.sample(wordlist, 7)
    LOGGER.info(f"All words: {', '.join(words_to_hit)}")

    CONFIG = {"words_to_hit": words_to_hit, "words_to_avoid": words_to_avoid, "n": n}

    glove_embedding_path = os.path.join("..", "data", "word_embeddings", "glove", "glove.6B.300d.txt")
    glove_solver = GloveSolver(**CONFIG, embedding_path=glove_embedding_path, threshold=0.5).build()
    log_solutions(glove_solver.solve(algorithm))

    for _ in range(10):
        t = random.uniform(0, 0.5)
        LOGGER.info(f"New Threshold: {t}")
        glove_solver.threshold = t
        glove_solver.words_to_hit = random.sample(wordlist, 7)
        LOGGER.info(f"All words: {', '.join(words_to_hit)}")
        log_solutions(glove_solver.solve(algorithm))

    # log_solutions(run_glove_solver(algorithm, .05))
    # log_solutions(run_postspec_solver(algorithm, .35))

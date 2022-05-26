import os
import time
import sys
import random

sys.path.append(os.path.abspath(os.path.join('..')))

from codenames.wordlist import WordListBuilder
from bot.algorithms import MeanIndividualDistance, SummedNearestNeighbour
from bot.distance import Cosine, DotProduct
from bot.solver import SolverBuilder
from bot.utils import initialise_logger, log_solutions


def get_words(words):
    random.shuffle(words)
    negative = words[7:10]
    positive = words[:6]
    return positive, negative


def manual_input():
    embedding_path = os.path.join("..", "data", "word_embeddings", "fasttext", "embeddings-200k.txt")
    builder = SolverBuilder.with_embeddings(embedding_path, "fasttext")
    solver = builder.build(SummedNearestNeighbour, 0.2, Cosine)
    words = input("Enter some words to connect with a space in between each...")
    while words:
        print("Thinking...")
        words = words.split(' ')
        log_solutions(solver.solve(words, n=50))
        time.sleep(0.5)
        words = input("Enter some more words to connect with a space in between each or press enter to escape...")


def random_task():
    path_to_word_list = os.path.join("..", "data", "wordlist-eng.txt")
    wordlist_builder = WordListBuilder(path_to_word_list)
    p_path = ["..", "data", "word_embeddings", "glove", "glove.6B.300d.txt"]
    embedding_path = os.path.join(*p_path)
    solverbuilder = SolverBuilder.with_embeddings(embedding_path, "glove")
    for _ in range(20):
        wordlist = wordlist_builder.build().wordlist
        words_to_hit, words_to_avoid = get_words(wordlist)
        logger.info(f"Positive Words: {', '.join(words_to_hit)}\nNegative Words: {', '.join(words_to_avoid)}")

        solver = solverbuilder.build(MeanIndividualDistance, 0.2, Cosine)
        solver.threshold = .25
        log_solutions(solver.solve(words_to_hit=words_to_hit, n=10))


if __name__ == "__main__":
    logger = initialise_logger()
    conf_path = os.path.join("..", "data", "params.csv")
    manual_input()

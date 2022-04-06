import os
import random

from codenames.wordlist import WordListBuilder
from solver.algorithms import SummedNearestNeighbour, MeanIndividualDistance
from solver.distance import DotProduct, Cosine
from solver.solver import SolverBuilder
from solver.utils import initialise_logger, log_solutions


def get_words(words):
    random.shuffle(words)
    negative = words[20:]
    positive = words[:7]
    return positive, negative


if __name__ == "__main__":

    GLOVE_PATH = os.path.join("..", "data", "word_embeddings", "glove", "glove.6B.300d.txt")

    p_path = ["..", "data", "word_embeddings", "post-specialized embeddings", "postspec", "glove_postspec.txt"]
    POSTSPEC_PATH = os.path.join(*p_path)

    w_path = ["..", "data", "word_embeddings", "wordnetemb", "embedding_cleaned.txt"]
    WORDNET_PATH = os.path.join(*w_path)

    b_path = ["..", "data", "word_embeddings", "bert", "embedding.txt"]
    BERT_PATH = os.path.join(*b_path)

    conf_path = os.path.join("..", "data", "params.csv")

    path_to_word_list = os.path.join("..", "data", "wordlist-eng.txt")
    wordlist = WordListBuilder(path_to_word_list).build().wordlist
    words_to_hit, words_to_avoid = get_words(wordlist)

    LOGGER = initialise_logger()

    strategies = ['risky', 'quite_risky', 'moderate', 'quite_conservative', 'conservative']
    distance_metrics = [DotProduct, Cosine]
    algorithms = [MeanIndividualDistance, SummedNearestNeighbour]

    n = 10
    CONFIG = {"words_to_hit": words_to_hit, "n": n}

    if CONFIG.get("words_to_avoid"):
        LOGGER.info(f"Words to link: {', '.join(words_to_hit)}\nWords to avoid: {', '.join(words_to_avoid)}")
    else:
        LOGGER.info(f"Words to link: {', '.join(words_to_hit)}\n")

    builder_methods = {
        "with_postspec": POSTSPEC_PATH,
        "with_wordnet": WORDNET_PATH,
        "with_glove": GLOVE_PATH,
        "with_bert": BERT_PATH
    }

    for method in builder_methods.keys():
        solverbuilder = getattr(SolverBuilder, method)(embedding_path=builder_methods[method])
        for algorithm in algorithms:
            for distance in distance_metrics:
                try:
                    LOGGER.info(f"Running {method.split('_')[1]} with {algorithm.__name__} and {distance.__name__} as distance metric.")
                    solverbuilder.algorithm = algorithm
                    solverbuilder.distance_metric = distance
                    solver = solverbuilder.build(conf_path, algorithm, distance, 'moderate')
                    log_solutions(solver.solve(words_to_hit=words_to_hit, n=n))
                except Exception as e:
                    LOGGER.error(f"Errored with exception: {e}")

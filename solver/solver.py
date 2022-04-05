import logging

import numpy as np
from tqdm import tqdm

from solver.config import GloveThreshold, PostSpecThreshold, WordNetThreshold, StaticBertThreshold
from solver.distance import DotProduct
from solver.utils import get_embeddings_glove_style


class Solver:
    def __init__(self, words_to_hit: list, words_to_avoid: list, model, n: int, threshold: float, distance_metric):
        """General Codenames Solver Class

        :param words_to_hit:
        :param words_to_avoid:
        :param model:
        :param n:
        :param strategy: Either risky, quite_risky, moderate, quite_conservative, conservative
        """
        self.words_to_hit = words_to_hit
        self.words_to_avoid = words_to_avoid
        self.model = model
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.n = n

    def solve(self, algorithm) -> list:
        """Takes algorithm object and gives prediction for best clues to link your words and avoid words that are not
        yours.

        :param algorithm: A solver.algorithm object that contains and solve method.
        :return: List of self.n Guess objects.
        """
        return algorithm(model=self.model,
                         words_to_hit=self.words_to_hit,
                         words_to_avoid=self.words_to_avoid,
                         n=self.n,
                         threshold=self.threshold,
                         distance_metric=self.distance_metric
                         ).solve()


class SolverBuilder:
    def __init__(self, words_to_hit: list, words_to_avoid: list = None, distance_metric=DotProduct, n: int = 5,
                 threshold: float = 0.3):
        self.words_to_hit = words_to_hit
        self.words_to_avoid = words_to_avoid
        self.distance_metric = distance_metric
        self.n = n
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def _build_language_model(self):
        raise NotImplementedError

    def build(self) -> Solver:
        model = self._build_language_model()
        return Solver(model=model,
                      words_to_hit=self.words_to_hit,
                      words_to_avoid=self.words_to_avoid,
                      distance_metric=self.distance_metric,
                      threshold=self.threshold,
                      n=self.n)


class GloveSolver(SolverBuilder):
    def __init__(self, embedding_path: str, words_to_hit: list, words_to_avoid: list = None, distance_metric=DotProduct,
                 n: int = 5, strategy: str = 'moderate'):
        super().__init__(words_to_hit, words_to_avoid, distance_metric, n)
        self.threshold = getattr(GloveThreshold, strategy)
        self.strategy = strategy
        self.embedding_path = embedding_path
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using {self.strategy} strategy with threshold: {self.threshold}")

    def _build_language_model(self) -> dict:
        self.logger.info("Loading GloVe embeddings...")
        embeddings = get_embeddings_glove_style(self.embedding_path)
        self.logger.info("GloVe embeddings loaded.")
        return embeddings


class PostSpecSolver(SolverBuilder):
    def __init__(self, embedding_path: str, words_to_hit: list, words_to_avoid: list = None, distance_metric=DotProduct,
                 n: int = 5, strategy: str = 'moderate'):
        # See https://github.com/cambridgeltl/adversarial-postspec
        super().__init__(words_to_hit, words_to_avoid, distance_metric, n)
        self.threshold = getattr(PostSpecThreshold, strategy)
        self.strategy = strategy
        self.embedding_path = embedding_path
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using {self.strategy} strategy with threshold: {self.threshold}")

    def _build_language_model(self) -> dict:
        self.logger.info("Loading PostSpec embeddings...")
        embeddings = {}
        with open(self.embedding_path, "r") as file:
            for line in tqdm(file):
                split_line = line.split()
                word = split_line[0].split('_')
                if word[0] == 'en':
                    word = word[1]
                    embedding = np.array(split_line[1:], dtype=np.float64)
                    embeddings[word] = embedding
        self.logger.info("PostSpec embeddings loaded.")
        return embeddings


class WordNetSolver(SolverBuilder):
    def __init__(self, embedding_path: str, words_to_hit: list, words_to_avoid: list = None, distance_metric=DotProduct,
                 n: int = 5, strategy: str = 'moderate'):
        # See https://github.com/asoroa/ukb
        super().__init__(words_to_hit, words_to_avoid, distance_metric, n)
        self.threshold = getattr(WordNetThreshold, strategy)
        self.strategy = strategy
        self.embedding_path = embedding_path
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using {self.strategy} strategy with threshold: {self.threshold}")

    def _build_language_model(self) -> dict:
        self.logger.info("Loading WordNet embeddings...")
        embeddings = get_embeddings_glove_style(self.embedding_path)
        self.logger.info("WordNet embeddings loaded.")
        return embeddings


class StaticBertSolver(SolverBuilder):
    def __init__(self, embedding_path: str, words_to_hit: list, words_to_avoid: list = None, distance_metric=DotProduct,
                 n: int = 5, strategy: str = 'moderate'):
        # See https://github.com/asoroa/ukb
        super().__init__(words_to_hit, words_to_avoid, distance_metric, n)
        self.threshold = getattr(StaticBertThreshold, strategy)
        self.strategy = strategy
        self.embedding_path = embedding_path
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using {self.strategy} strategy with threshold: {self.threshold}")

    def _build_language_model(self) -> dict:
        self.logger.info("Loading Static BERT embeddings...")
        embeddings = get_embeddings_glove_style(self.embedding_path)
        self.logger.info("Static BERT embeddings loaded.")
        return embeddings

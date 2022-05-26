import logging
from typing import Type, Callable

from bot.algorithms import MeanIndividualDistance, CodeNamesSolverAlgorithm
from bot.distance import DotProduct, Cosine
from bot.threshold import Threshold
from bot.utils import get_embeddings_glove_style, EmbeddingsDataLoader


class SolverBuilder:
    def __init__(self, model=None, method: str = ''):
        self.model = model
        self.method = method
        self.logger = logging.getLogger(__name__)

    def build(self, algorithm: Type[CodeNamesSolverAlgorithm] = MeanIndividualDistance, threshold: float = 0.3,
              distance_metric=Cosine, strategy: str = None, conf_path: str = None) -> CodeNamesSolverAlgorithm:
        """Base builder class, main interface for solving Codenames. Typically, built with one of class methods.

        :param conf_path: Path to conf that contains .csv with cols for threshold, algorithm, distance, strategy, model
        :param algorithm: Class of type CodeNamesSolverAlgorithm that allows bot to solve
        :param distance_metric: Class for distance found in distance.py. Select from (Cosine, DotProduct, Euclidian)
        :param strategy: Str from risky, quite_risky, moderate, quite_conservative, conservative. Controls how close
        words need to be connected. If not included, defaults to threshold that equates to moderate
        :param threshold: Akin to strategy, controls how close words need to be connected
        :return: CodeNamesSolverAlgorithm class that can solve for search words
        """

        if strategy:
            args = {
                "model": self.method,
                "algorithm": algorithm,
                "strategy": strategy,
                "distance": distance_metric,
                "conf_path": conf_path
            }
            threshold = Threshold.from_config(**args).threshold

        return algorithm(model=self.model, threshold=threshold, distance_metric=distance_metric)

    @classmethod
    def with_embeddings(cls, embedding_path: str, name: str, embeddings_parser: Callable = get_embeddings_glove_style):
        """Core component of solver. Feed a path to local embeddings.
         Recommended options:
         - Postspec: https://github.com/cambridgeltl/adversarial-postspec
         - BERT: https://openreview.net/attachment?id=SJg3T2EFvr&name=original_pdf
                https://papers.nips.cc/paper/2019/file/159c1ffe5b61b41b3c4d8f4c2150f6c4-Paper.pdf
        - WordNet:
        - GloVe:
        - Definitional:
        - Fasttext:

        :param embedding_path: Path to embeddings
        :param name: Name used for logging and checking thresholds
        :param embeddings_parser: Function to load embeddings. Select from:
            get_embeddings_glove_style, get_embeddings_postspec_style
        :return: SolverBuilder
        """
        embeddings = EmbeddingsDataLoader(embedding_path).get_embeddings(embeddings_parser, name)
        return cls(embeddings, name.lower())

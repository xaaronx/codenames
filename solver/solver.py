import logging
from typing import Type

from solver.algorithms import MeanIndividualDistance, CodeNamesSolverAlgorithm
from solver.distance import DotProduct
from solver.threshold import Threshold
from solver.utils import get_embeddings_glove_style, get_embeddings_postspec_style, EmbeddingsDataLoader


class SolverBuilder:
    def __init__(self, model=None, method: str = ''):
        self.model = model
        self.method = method
        self.logger = logging.getLogger(__name__)

    def build(self, conf_path: str, algorithm: Type[CodeNamesSolverAlgorithm] = MeanIndividualDistance,
              distance_metric=DotProduct, strategy: str = "moderate") -> CodeNamesSolverAlgorithm:
        """Base builder class, main interface for solving Codenames. Typically, built with one of class methods.

        :param conf_path: Path to conf that contains .csv with cols for threshold, algorithm, distance, strategy, model
        :param algorithm: Class of type CodeNamesSolverAlgorithm that allows solver to solve
        :param distance_metric: Class for distance found in distance.py. Select from (Cosine, DotProduct, Euclidian)
        :param strategy: Str from risky, quite_risky, moderate, quite_conservative, conservative. Controls how close
        words need to be connected
        :return: CodeNamesSolverAlgorithm class that can solve for search words
        """

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
    def with_glove(cls, embedding_path: str):
        """GloVe model based solver. Associative word embeddings.

        :param embedding_path: Path to glove embeddings
        :return: SolverBuilder with model now built
        """
        embeddings = EmbeddingsDataLoader(embedding_path).get_embeddings(get_embeddings_glove_style, "GloVe")
        return cls(embeddings, "glove")

    @classmethod
    def with_postspec(cls, embedding_path: str):
        """Post specialised word embeddings with help of wordnet.
        See: https://github.com/cambridgeltl/adversarial-postspec

        :param embedding_path: Path to postspec embeddings
        :return: SolverBuilder with model now built
        """
        embeddings = EmbeddingsDataLoader(embedding_path).get_embeddings(get_embeddings_postspec_style, "PostSpec")
        return cls(embeddings, "postspec")

    @classmethod
    def with_wordnet(cls, embedding_path: str):
        """WordNet converted into embeddings. More direct relational word connections.

        :param embedding_path: Path to wordnet embeddings
        :return: SolverBuilder with model now built
        """
        embeddings = EmbeddingsDataLoader(embedding_path).get_embeddings(get_embeddings_glove_style, "WordNet")
        return cls(embeddings, "wordnet")

    @classmethod
    def with_bert(cls, embedding_path: str):
        """Static BERT based word embeddings. Typically, not effective because context matters. Also, we can't
        establish context easily with single words!

        :param embedding_path: Path to BERT embeddings
        :return: SolverBuilder with model now built
        """
        embeddings = EmbeddingsDataLoader(embedding_path).get_embeddings(get_embeddings_glove_style, "BERT")
        return cls(embeddings, "bert")

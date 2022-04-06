import logging

from solver.algorithms import MeanIndividualDistance, CodeNamesSolverAlgorithm
from solver.distance import DotProduct
from solver.threshold import Threshold
from solver.utils import get_embeddings_glove_style, get_embeddings_postspec_style, EmbeddingsDataLoader


class SolverBuilder:
    def __init__(self, model=None, method: str = ''):
        self.model = model
        self.method = method
        self.logger = logging.getLogger(__name__)

    def build(self, conf_path: str, algorithm=MeanIndividualDistance, distance_metric=DotProduct,
              strategy: str = "moderate") -> CodeNamesSolverAlgorithm:

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
        embeddings = EmbeddingsDataLoader(embedding_path).get_embeddings(get_embeddings_glove_style, "GloVe")
        return cls(embeddings, "glove")

    @classmethod
    def with_postspec(cls, embedding_path: str):
        embeddings = EmbeddingsDataLoader(embedding_path).get_embeddings(get_embeddings_postspec_style, "PostSpec")
        return cls(embeddings, "postspec")

    @classmethod
    def with_wordnet(cls, embedding_path: str):
        embeddings = EmbeddingsDataLoader(embedding_path).get_embeddings(get_embeddings_glove_style, "WordNet")
        return cls(embeddings, "wordnet")

    @classmethod
    def with_bert(cls, embedding_path: str):
        embeddings = EmbeddingsDataLoader(embedding_path).get_embeddings(get_embeddings_glove_style, "BERT")
        return cls(embeddings, "bert")

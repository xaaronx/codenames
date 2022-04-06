import logging

from solver.algorithms import MeanIndividualDistance, CodeNamesSolverAlgorithm
from solver.distance import DotProduct
from solver.threshold import Threshold
from solver.utils import get_embeddings_glove_style, get_embeddings_postspec_style, EmbeddingsDataLoader


class SolverBuilder:
    def __init__(self, model=None, algorithm=MeanIndividualDistance, distance_metric=DotProduct, threshold: float = .3):
        self.model = model
        self.algorithm = algorithm
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def build(self) -> CodeNamesSolverAlgorithm:
        return self.algorithm(model=self.model,
                              threshold=self.threshold,
                              distance_metric=self.distance_metric)

    @classmethod
    def with_glove(cls, embedding_path: str, algorithm=MeanIndividualDistance, distance_metric=DotProduct,
                   strategy: str = 'moderate', conf_path: str = None):
        threshold = Threshold.from_config("glove", algorithm, strategy, distance_metric, conf_path).threshold
        embeddings = EmbeddingsDataLoader(embedding_path).get_embeddings(get_embeddings_glove_style, "GloVe")
        return cls(embeddings, algorithm, distance_metric, threshold)

    @classmethod
    def with_postspec(cls, embedding_path: str, algorithm=MeanIndividualDistance, distance_metric=DotProduct,
                      strategy: str = 'moderate', conf_path: str = None):
        threshold = Threshold.from_config("postspec", algorithm, strategy, distance_metric, conf_path).threshold
        embeddings = EmbeddingsDataLoader(embedding_path).get_embeddings(get_embeddings_postspec_style, "PostSpec")
        return cls(embeddings, algorithm, distance_metric, threshold)

    @classmethod
    def with_wordnet(cls, embedding_path: str, algorithm=MeanIndividualDistance, distance_metric=DotProduct,
                     strategy: str = 'moderate', conf_path: str = None):
        threshold = Threshold.from_config("wordnet", algorithm, strategy, distance_metric, conf_path).threshold
        embeddings = EmbeddingsDataLoader(embedding_path).get_embeddings(get_embeddings_glove_style, "BERT")
        return cls(embeddings, algorithm, distance_metric, threshold)

    @classmethod
    def with_bert(cls, embedding_path: str, algorithm=MeanIndividualDistance, distance_metric=DotProduct,
                  strategy: str = 'moderate', conf_path: str = None):
        threshold = Threshold.from_config("bert", algorithm, strategy, distance_metric, conf_path).threshold
        embeddings = EmbeddingsDataLoader(embedding_path).get_embeddings(get_embeddings_glove_style, "BERT")
        return cls(embeddings, algorithm, distance_metric, threshold)

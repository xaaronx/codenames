import logging

import numpy as np
from tqdm import tqdm


class EmbeddingsDataLoader:
    def __init__(self, fpath: str):
        self.logger = logging.getLogger(__name__)
        self.fpath = fpath

    def get_embeddings(self, func, name: str):
        self.logger.info(f"Loading {name} embeddings...")
        embeddings = func(self.fpath)
        self.logger.info(f"{name} embeddings loaded.")
        return embeddings


def remove_keys_from_dict(dictionary: dict, keys_to_remove: list) -> dict:
    dictionary_copy = dictionary.copy()
    for key in keys_to_remove:
        dictionary_copy.pop(key, None)
    return dictionary_copy


def get_top_n_sorted(values: np.array, n: int = 5) -> np.array:
    top_n_items = np.argpartition(values, -n)[-n:]
    indices = top_n_items[np.argsort(-values[top_n_items])]
    return indices


def np_cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embeddings_glove_style(path: str):
    embeddings = {}
    with open(path, "r") as file:
        for line in tqdm(file):
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            embeddings[word] = embedding
    return embeddings


def get_embeddings_postspec_style(self) -> dict:
    embeddings = {}
    with open(self.embedding_path, "r") as file:
        for line in tqdm(file):
            split_line = line.split()
            word = split_line[0].split('_')
            if word[0] == 'en':
                word = word[1]
                embedding = np.array(split_line[1:], dtype=np.float64)
                embeddings[word] = embedding
    return embeddings


def initialise_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(message)8s')
    return logger


def log_solutions(solutions):
    logger = logging.getLogger(__name__)
    dash = '-' * 80
    formatting = '{:<20s}{:<30s}{:<40}'
    for i, s in enumerate(solutions):
        if i == 0:
            logger.info(dash)
            logger.info(formatting.format("Clue", "Score", "Linked Words"))
            logger.info(dash)
        else:
            logger.info(formatting.format(s.clue, str(round(s.score, 3)), ', '.join(s.linked_words)))
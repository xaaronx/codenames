import numpy as np
from tqdm import tqdm


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

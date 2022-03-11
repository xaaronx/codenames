import numpy as np


def remove_keys_from_dict(dictionary, keys_to_remove):
    dictionary_copy = dictionary.copy()
    for key in keys_to_remove:
        dictionary_copy.pop(key, None)
    return dictionary_copy


def get_top_n_sorted(values: np.array, n: int = 5):
    top_n_items = np.argpartition(values, -n)[-n:]
    indices = top_n_items[np.argsort(-values[top_n_items])]
    return indices

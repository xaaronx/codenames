from collections.abc import Iterable
import numpy as np


def flatten(list_of_lists):
    """https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    :param list_of_lists
    :return: flattened list
    """
    for element in list_of_lists:
        if isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
            yield from flatten(element)
        else:
            yield element


def matches_as_set(array: np.array, value: int):
    matches = np.where(array == value)
    return set(zip(matches[0].tolist(), matches[1].tolist()))

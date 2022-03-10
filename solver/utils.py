def remove_keys_from_dict(dictionary, keys_to_remove):
    for key in keys_to_remove:
        dictionary.pop(key, None)
    return dictionary

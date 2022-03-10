import numpy as np
from sklearn.neighbors import NearestNeighbors

from solver.utils import remove_keys_from_dict


class Solver:
    def __init__(self, words_to_hit: list, words_to_avoid: list, embeddings: dict):
        self.words_to_hit = words_to_hit
        self.words_to_avoid = words_to_avoid
        self.embeddings = embeddings

    def nearest_neighbor_to_average_of_words(self):
        embeddings_of_words_to_hit = np.array([self.embeddings[word] for word in self.words_to_hit])
        target_vector = np.mean(embeddings_of_words_to_hit, axis=0)
        temp_embeddings = remove_keys_from_dict(self.embeddings, self.words_to_hit)
        embeddings_as_array = np.array(list(temp_embeddings.values()))
        similarities = np.dot(target_vector, embeddings_as_array.T)
        word_index = np.argmax(similarities)
        return list(temp_embeddings.keys())[word_index]


class SolverBuilderGlove:
    def __init__(self, words_to_hit: list, words_to_avoid: list, embedding_path: str):
        self.words_to_hit = words_to_hit
        self.words_to_avoid = words_to_avoid
        self.embedding_path = embedding_path

    def persist_embeddings(self):
        embeddings = {}
        with open(self.embedding_path, "r") as file:
            for line in file:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                embeddings[word] = embedding
        return embeddings

    @staticmethod
    def build_nearest_neighbour_index(embeddings):
        return NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(embeddings)

    def build(self):
        embeddings = self.persist_embeddings()
        return Solver(words_to_hit=self.words_to_hit,
                      words_to_avoid=self.words_to_avoid,
                      embeddings=embeddings)

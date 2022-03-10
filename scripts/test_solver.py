import os

from solver.solver import SolverBuilderGlove

if __name__ == "__main__":
    words_to_avoid = ["dog", "cat", "elephant", "France"]
    words_to_hit = ["cheese", "milk", "butter"]

    solver = SolverBuilderGlove(words_to_hit=words_to_hit,
                                words_to_avoid=words_to_avoid,
                                embedding_path=os.path.join("..", "data", "word_embeddings", "glove.6B.50d.txt")
                                ).build()

    print(solver.nearest_neighbor_to_average_of_words())

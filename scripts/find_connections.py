import os

from bot.algorithms import SummedNearestNeighbour
from bot.solver import WordNetSolver

if __name__ == "__main__":

    n = 10
    words_to_hit = ["rag", "criticise", "knock"]
    CONFIG = {"words_to_hit": words_to_hit, "n": n}
    w_path = ["..", "data", "word_embeddings", "wordnetemb", "embedding_cleaned.txt"]
    wordnet_embedding_path = os.path.join(*w_path)
    wordnet_solver = WordNetSolver(**CONFIG, embedding_path=wordnet_embedding_path).build()
    model = wordnet_solver.model
    print(SummedNearestNeighbour(model=model, words_to_hit=words_to_hit, n=n, threshold=0.5)._compute(words_to_hit))
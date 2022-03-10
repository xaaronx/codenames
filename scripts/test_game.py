import os
import numpy as np

from game.game import GameBuilder

if __name__ == "__main__":
    word_list_file_name = "wordlist-eng.txt"
    word_list_path = os.path.join("..", "data", word_list_file_name)
    game = GameBuilder(word_path=word_list_path).build()


import random

from codenames.board import Board
import numpy as np

from codenames.colours import Colour
from codenames.utils import flatten, matches_as_set
from codenames.wordlist import WordListBuilder


class Game:
    def __init__(self, words: np.array, word_colours: np.array, state: np.array):
        self.words = words
        self.answers = word_colours
        self.state = state
        self.turn = 0

    def _is_finished(self):
        player_1_won = self._check_if_player_has_won(player_id=1)
        player_2_won = self._check_if_player_has_won(player_id=2)

    def _check_if_player_has_won(self, player_id: int):
        guesses = matches_as_set(self.state, player_id)
        answers = matches_as_set(self.answers, player_id)

        if answers.issubset(guesses):
            return True
        elif self.landed_on_bomb():
            return
        else:
            return


class GameBuilder(Board):
    def __init__(self, word_path: str):
        super().__init__()
        self.word_path = word_path
        self.n_words = self.x * self.y
        self.state = np.zeros(shape=(self.x, self.y))

    def build(self):
        game_words = WordListBuilder(path=self.word_path, n_words=self.n_words).build().wordlist
        game_words_array = np.array(game_words).reshape(self.x, self.y)
        word_colours = self._get_word_colours()
        return Game(words=game_words_array, word_colours=word_colours, state=self.state)

    def _create_colour_stack(self):
        return list(flatten(
            [
                [Colour.BLACK for _ in range(self.n_board_words_black)],
                [Colour.GREY for _ in range(self.n_board_words_grey)],
                [Colour.BLUE for _ in range(self.n_board_words_blue)],
                [Colour.RED for _ in range(self.n_board_words_red)]
            ]
        ))

    def _get_word_colours(self):
        colour_stack = self._create_colour_stack()
        random.shuffle(colour_stack)
        word_colours = np.zeros(shape=(self.x, self.y))
        for x in range(self.x):
            for y in range(self.y):
                colour = colour_stack.pop()
                word_colours[x, y] = colour

        return word_colours

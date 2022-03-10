class Board:
    def __init__(self,
                 board_size_x: int = 5,
                 board_size_y: int = 5,
                 n_board_words_red: int = 9,
                 n_board_words_blue: int = 8,
                 n_board_words_grey: int = 7,
                 n_board_words_black: int = 1,
                 ):

        self.x = board_size_x
        self.y = board_size_y
        self.n_board_words_blue = n_board_words_blue
        self.n_board_words_red = n_board_words_red
        self.n_board_words_grey = n_board_words_grey
        self.n_board_words_black = n_board_words_black
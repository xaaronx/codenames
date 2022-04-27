from dataclasses import field, dataclass


@dataclass
class Guess:
    clue: str
    similarity_score: float
    linked_words: list
    num_words_linked: int = field(init=False)

    def __post_init__(self):
        self.num_words_linked = self.get_num_words_linked()

    def get_num_words_linked(self) -> int:
        return len(self.linked_words)

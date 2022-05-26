from dataclasses import field, dataclass, asdict
from typing import Optional


@dataclass
class Guess:
    clue: str
    similarity: float
    linked_words: list
    score: Optional[float] = .0
    num_words_linked: int = field(init=False)

    def __post_init__(self):
        self.num_words_linked = self.get_num_words_linked()

    def get_num_words_linked(self) -> int:
        return len(self.linked_words)

    def as_dict(self):
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}

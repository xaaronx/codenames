import random


class WordList:
    def __init__(self, wordlist):
        self.wordlist = wordlist


class WordListBuilder:
    def __init__(self, path: str, n_words: int = 25, lower: bool = True):
        self.n_words = n_words
        self.path = path
        self.lower = lower

    def get_full_word_list(self):
        all_words = self._persist_words()
        return WordList(wordlist=all_words)

    def build(self):
        all_words = self._persist_words()
        wordlist = random.sample(all_words, self.n_words)
        return WordList(wordlist=wordlist)

    def _persist_words(self):
        all_words = open(self.path).read().split()
        if self.lower:
            return [word.lower() for word in all_words]
        return all_words

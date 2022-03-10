import random


class WordList:
    def __init__(self, wordlist):
        self.wordlist = wordlist


class WordListBuilder:
    def __init__(self, path: str, n_words: int = 25, lower: bool = True):
        self.n_words = n_words
        self.path = path
        self.lower = lower

    def build(self):
        all_words = open(self.path).read().split()
        if self.lower:
            all_words = [word.lower() for word in all_words]
        wordlist = random.sample(all_words, self.n_words)
        return WordList(wordlist=wordlist)

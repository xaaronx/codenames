# Codenames Game + Solver

Codenames is a game where players need to connect as many words as possible to a single clue.

The game makes for a fascinating NLP challenge - it can be solved in many ways and can also help
to teach many foundational NLP ideas.

Typically, it is played between 2 teams of 2 where each team has has a spymaster and an operative.
The spymaster needs to think up a word to connect their teams words. They may only use one word and a number for the 
number of words that they're intending to connect (e.g. DOG 3). The operative is then tasked with identifying
the words on the board that link to dog.

---

## Solver

The solver has been designed to allow language models and solving algorithms to be easily slotted in to the Solver object.

Algorithms typically have a few methods that power a **solve** method.
Solver builders have a few methods that power a **build** method that.

#### Still to do:

- **Efficiency**
  - FAISS or similar to calculate nearest neighbors more efficiently
  - Vectorize rather than finding similarities per guess
  - Consider trimming embedding wordlists
- **Performance**
  - GPT-3?
  - Sentence Transformers?
- **Documentation**
  - Docstrings
- **Functionality**
  - Bomb word as well as words to avoid
- **Validation**
  - Test against dataset?
  - Scrape a dataset?
- **Algorithms**:
  - Add word concatenation and checking (eg. 'water', 'gate' -> 'watergate')
  - Get thresholds to work smarter (they're random and dont generalise well across algorithms or solvers)

---

## Game

#### TODO: 
- Lots :)
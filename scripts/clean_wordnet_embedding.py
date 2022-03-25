import os

from tqdm import tqdm

if __name__ == "__main__":
    # common_words = set(
    #     [line.strip() for line in open(os.path.join("..", "..", "data", "google-10000-english.txt"), "r")])

    common_words = set(
        [line.strip().split()[0] for line in open(os.path.join("..", "data", "word_embeddings", "glove", "glove.6B.50d.txt"), "r")])

    new_file = []
    with open(os.path.join("..", "data", "word_embeddings", "wordnetemb", "embedding.emb"), "r") as file:
        next(file)
        for line in tqdm(file):
            split_line = line.strip().split()
            word = split_line[0]
            if word.isalnum() and word in common_words:
                new_file.append(line.strip())

    outpath = os.path.join("..", "data", "word_embeddings", "wordnetemb", "embedding_cleaned.txt")
    with open(outpath, "w") as outfile:
        outfile.write("\n".join(new_file))

from collections import defaultdict

import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")


class Vocabulary:
    def __init__(self):
        self.word2idx = defaultdict(lambda: self.word2idx["<unk>"])
        self.word2idx["<pad>"] = 0
        self.word2idx["<unk>"] = 1
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.idx = 2

    def add_sentence(self, sentence):
        for word in word_tokenize(sentence):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

    def encode(self, sentence):
        return [self.word2idx[word] for word in word_tokenize(sentence)]

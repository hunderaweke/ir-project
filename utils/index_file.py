import json
from typing import List
from collections import defaultdict
from preprocessors.tokenizer import Tokenizer
from preprocessors.stemmer import MainStemmer


class IndexFileBuilder:
    words = set()
    index = defaultdict(lambda: {"doc_ids": [], "positions": defaultdict(list)})

    @classmethod
    def build(cls, documents: List[str]):
        tokens = {
            i: MainStemmer.stem(Tokenizer.tokenize(word))
            for i, word in enumerate(documents)
        }
        for doc_id, words in tokens.items():
            for pos, token in enumerate(words):
                cls.index[token]["doc_ids"].append(doc_id)
                cls.index[token]["positions"][doc_id].append(pos)
                cls.words.add(token)
        return cls.index

    @classmethod
    def create_index_file(cls):
        with open("inverted_index.json", "w") as file:
            json.dump(cls.index, file)

    @classmethod
    def create_posting_file(cls):
        cls.words = sorted(cls.words)
        with open("posting_file.json", "w+") as file:
            json.dump(cls.words, file)

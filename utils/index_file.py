from typing import List
from collections import defaultdict
from preprocessors.tokenizer import Tokenizer
from preprocessors.stemmer import MainStemmer


class IndexFileBuilder:
    words = []
    indices = defaultdict(set)
    posting = defaultdict(list)

    @classmethod
    def build(cls, documents: List[str]):
        tokens = {
            i: MainStemmer.stem(Tokenizer.tokenize(word))
            for i, word in enumerate(documents)
        }
        for doc_id, words in tokens.items():
            for pos, token in enumerate(words):
                cls.posting[token].append((doc_id, pos))
                cls.indices[token].add(doc_id)
        return cls.indices, cls.posting

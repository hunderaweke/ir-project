import re
from typing import List

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}


class Tokenizer:
    @classmethod
    def filter(cls, tokens: List[str]):
        filtered_tokens = []
        for token in tokens:
            if token in STOP_WORDS:
                continue
            filtered_tokens.append(token)
        return filtered_tokens

    @classmethod
    def tokenize(cls, word: str) -> List[str]:
        tokens = re.findall(pattern=r"\b\w+\b", string=word.lower())
        tokens = cls.filter(tokens)
        return tokens


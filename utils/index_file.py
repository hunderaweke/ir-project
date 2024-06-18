import re
from typing import List


class Tokenizer:
    @classmethod
    def tokenize(cls, word: str) -> List[str]:
        tokens = re.findall(pattern=r"\b\w+\b", string=word.lower())
        return tokens

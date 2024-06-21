from preprocessors.tokenizer import Tokenizer
from preprocessors.stemmer import MainStemmer
from utils.index_file import IndexFileBuilder

documents = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick red fox jumps over the sleepy cat.",
    "A fast blue fox leaps over the tired dog.",
    "Cunning foxes are jumping over a lazy hound.",
    "A group of lazy dogs were being jumped over by quick foxes.",
    "The tired fox quickly leapt over the sleeping cat.",
    "Several brown foxes have jumped over lazy dogs.",
    "A quick fox will jump over the lazy dog.",
    "The lazy dog was jumped over by a quick fox.",
    "Quick brown foxes are leaping over lazy dogs.",
    "The sleepy cat was being jumped over by the fast fox.",
]

print(IndexFileBuilder.build(documents))

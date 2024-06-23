import os

from pathlib import Path
from preprocessors.tokenizer import Tokenizer
from preprocessors.stemmer import MainStemmer, PorterStemmer
from ranker.vector_space import TfIdfVectorizer
from utils.index_file import IndexFileBuilder

index_to_name = {}
CORPUS_PATH = Path.cwd() / "corpus"
documents = []
for i, file_name in enumerate(os.listdir(CORPUS_PATH)):
    index_to_name[i] = file_name
    with open(CORPUS_PATH / file_name, "r+") as file:
        documents.append(file.readline().rstrip())

IndexFileBuilder.build(documents)
IndexFileBuilder.create_index_file()
IndexFileBuilder.create_posting_file()
model = TfIdfVectorizer(len(documents))
model.calculate_tf_idf()
print("Welcome to Simple Information Retrieval System")
query = input("Please Enter a query for searching in the documents: ")
query = MainStemmer.stem(Tokenizer.tokenize(query))
print("Here is the rank of the documents for the given query.")
for i, indx in enumerate(model.rank_docs(query), start=1):
    print(i, index_to_name[indx])

import json
import math
from typing import List


class TfIdfVectorizer:
    def __init__(self, num_docs: int) -> None:
        with open("inverted_index.json", "r") as f:
            self.inverted_index = json.load(f)
        with open("posting_file.json", "r") as f:
            self.words = json.load(f)
        self.num_words = len(self.words)
        self.num_docs = num_docs
        self.term_doc_matrix = [[0] * (self.num_words) for _ in range(self.num_docs)]
        for indx, word in enumerate(self.words):
            for doc_id in self.inverted_index[word]["doc_ids"]:
                self.term_doc_matrix[doc_id][indx] += 1

    def calculate_tf_idf(self):
        self.doc_freq = [
            len(self.inverted_index[word]["doc_ids"]) for word in self.words
        ]
        self.idf = [
            math.log2((self.num_docs + 1) / (df + 1)) + 1 for df in self.doc_freq
        ]
        self.tf_idf_matrix = [[0] * self.num_words for _ in range(self.num_docs)]

        for term_indx in range(self.num_words):
            for doc_indx in range(self.num_docs):
                tf = self.term_doc_matrix[doc_indx][term_indx]
                idf = self.idf[term_indx]
                self.tf_idf_matrix[doc_indx][term_indx] = tf * idf
        return self.tf_idf_matrix

    def calc_query_vector(self, query):
        self.query_vector = [0] * self.num_words

        for term in query:
            if term in self.inverted_index:
                indx = self.words.index(term)
                self.query_vector[indx] += 1

        for indx in range(len(self.words)):
            self.query_vector[indx] *= self.idf[indx]
        return self.query_vector

    def rank_docs(self, query) -> List[float]:
        if not hasattr(self, "tf_idf_matrix"):
            self.calculate_tf_idf()
        if not hasattr(self, "query_vector"):
            self.calc_query_vector(query)
        vector_lengths = [
            math.sqrt(
                sum(self.tf_idf_matrix[doc_indx][i] ** 2 for i in range(self.num_words))
            )
            for doc_indx in range(self.num_docs)
        ]
        query_vector_length = math.sqrt(
            sum(self.query_vector[i] ** 2 for i in range(self.num_words))
        )
        dot_products = [
            sum(
                self.query_vector[i] * self.tf_idf_matrix[indx][i]
                for i in range(self.num_words)
            )
            for indx in range(self.num_docs)
        ]
        similarity_score = [
            (
                (
                    (dot_products[i] / (query_vector_length * vector_lengths[i]))
                    if (query_vector_length and vector_lengths[i])
                    else 0
                ),
                i,
            )
            for i in range(self.num_docs)
        ]
        similarity_score.sort(key=lambda x: x[0])
        return [indx for _, indx in similarity_score][::-1]

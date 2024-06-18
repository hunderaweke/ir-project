from preprocessors.tokenizer import Tokenizer
from preprocessors.stemmer import MainStemmer

i = """The quick brown fox jumps over the lazy dog. While the dog was sleeping, the fox was busy exploring the surroundings. 
This scenario shows how agile and swift foxes are. On the other hand, dogs are loyal and protective, always ready to defend their territory.
In this example, we see that both animals have their unique characteristics and behaviors. Such interactions between animals can be fascinating to observe."""
tokens = Tokenizer.tokenize(i)
print(MainStemmer.stem(tokens))
